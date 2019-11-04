import os
from collections import Counter

import numpy as np

import torch

from ignite.engine import Events, Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import RunningAverage, Metric, Loss
from ignite.handlers import TerminateOnNan
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, GradsScalarHandler


class AverageMetric(Metric):
    def reset(self):
        self._sums = Counter()
        self._num_examples = Counter()

    def update(self, output):
        for k, v in output.items():
            self._sums[k] += torch.sum(v)
            self._num_examples[k] += torch.numel(v)

    def compute(self):
        return {k: v / self._num_examples[k] for k, v in self._sums.items()}

    def completed(self, engine):
        engine.state.metrics = {**engine.state.metrics, **self.compute()}

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed)


class Trainer:
    _STEPS_PER_LOSS_WRITE = 10
    _STEPS_PER_GRAD_WRITE = 10

    def __init__(
            self,
            module,
            train_loss,
            valid_loss,
            test_metrics,
            train_loader,
            valid_loader,
            test_loader,
            opt,
            max_bad_valid_epochs,
            visualizer,
            writer,
            max_epochs,
            epochs_per_test,
            should_checkpoint_latest,
            should_checkpoint_best_valid,
            device
    ):
        self._module = module
        self._module.to(device)

        self._train_loss = train_loss
        self._valid_loss = valid_loss
        self._test_metrics = test_metrics

        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._test_loader = test_loader

        self._opt = opt

        self._max_epochs = max_epochs
        self._max_bad_valid_epochs = max_bad_valid_epochs
        self._best_valid_loss = float("inf")
        self._num_bad_valid_epochs = 0
        self._epochs_per_test = epochs_per_test

        self._should_checkpoint_best_valid = should_checkpoint_best_valid

        self._visualizer = visualizer

        self._writer = writer

        self._device = device

        self._trainer = Engine(self._train_batch)
        self._evaluator = Engine(self._validate_batch)
        self._tester = Engine(self._test_batch)

        AverageMetric().attach(self._trainer)
        AverageMetric().attach(self._evaluator)
        AverageMetric().attach(self._tester)

        ProgressBar(persist=True).attach(self._trainer, ["loss"])
        ProgressBar(persist=False, desc="Validating").attach(self._evaluator)
        ProgressBar(persist=False, desc="Testing").attach(self._tester)

        self._trainer.add_event_handler(Events.EPOCH_STARTED, lambda _: self._module.train())
        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_training_info)

        if should_checkpoint_latest:
            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._save_checkpoint("latest"))

        self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._validate)
        self._evaluator.add_event_handler(Events.EPOCH_STARTED, lambda _: self._module.eval())

        self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._test)
        self._tester.add_event_handler(Events.EPOCH_STARTED, lambda _: self._module.eval())

    def train(self):
        self._trainer.run(data=self._train_loader, max_epochs=self._max_epochs)

    def _train_batch(self, engine, batch):
        x, _ = batch # TODO: Potentially pass y also for genericity
        x = x.to(self._device)

        self._opt.zero_grad()
        loss = self._train_loss(self._module, x).mean()
        loss.backward()
        self._opt.step()

        return {"loss": loss}

    @torch.no_grad()
    def _test(self, engine):
        epoch = engine.state.epoch
        if (epoch - 1) % self._epochs_per_test == 0: # Test after first epoch
            state = self._tester.run(data=self._test_loader)

            for k, v in state.metrics.items():
                self._writer.write_scalar(f"test/{k}", v, global_step=engine.state.epoch)

            self._visualizer.visualize(self._module, epoch)

    def _test_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        return self._test_metrics(self._module, x)

    @torch.no_grad()
    def _validate(self, engine):
        state = self._evaluator.run(data=self._valid_loader)
        valid_loss = state.metrics["loss"]

        if valid_loss < self._best_valid_loss:
            print(f"Best validation loss {valid_loss} after epoch {engine.state.epoch}")
            self._num_bad_valid_epochs = 0
            self._best_valid_loss = valid_loss

            if self._should_checkpoint_best_valid:
                self._save_checkpoint(tag="best_valid")

        else:
            self._num_bad_valid_epochs += 1

            # We do this manually (i.e. don't use Ignite's early stopping) to permit
            # saving/resuming more easily
            if self._num_bad_valid_epochs > self._max_bad_valid_epochs:
                print(
                    f"No validation improvement after {self._num_bad_valid_epochs} epochs. Terminating."
                )
                self._trainer.terminate()

    def _validate_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        return {"loss": self._valid_loss(self._module, x)}

    def _log_training_info(self, engine):
        i = engine.state.iteration

        if i % self._STEPS_PER_LOSS_WRITE == 0:
            loss = engine.state.output["loss"]
            self._writer.write_scalar("train/loss", loss, global_step=i)

        if i % self._STEPS_PER_GRAD_WRITE == 0:
            self._writer.write_scalar("train/grad-norm", self._grad_norm(), global_step=i)

    def _grad_norm(self):
        norm = 0
        for param in self._module.parameters():
            if param.grad is not None:
                norm += param.grad.norm().item()**2
        return np.sqrt(norm)

    def _save_checkpoint(self, tag):
        # We do this manually (i.e. don't use Ignite's checkpointing) because
        # Ignite only allows saving objects, not scalars (e.g. the current epoch) 
        checkpoint = {
            "epoch": self._trainer.state.epoch,
            "iteration": self._trainer.state.iteration,
            "module_state_dict": self._module.state_dict(),
            "opt_state_dict": self._opt.state_dict(),
            "best_valid_loss": self._best_valid_loss,
            "num_bad_valid_epochs": self._num_bad_valid_epochs
        }

        self._writer.write_checkpoint(tag, checkpoint)
