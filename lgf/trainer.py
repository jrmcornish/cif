import os
from contextlib import suppress
from collections import Counter
import sys

import numpy as np

import torch
import torch.nn.utils

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
    _STEPS_PER_LR_WRITE = 10

    def __init__(
            self,

            module,
            device,

            train_metrics,
            train_loader,
            opt,
            lr_scheduler,
            max_epochs,
            max_grad_norm,

            test_metrics,
            test_loader,
            epochs_per_test,

            early_stopping,
            valid_loss,
            valid_loader,
            max_bad_valid_epochs,

            visualizer,

            writer,
            should_checkpoint_latest,
            should_checkpoint_best_valid
    ):
        self._module = module

        self._device = device

        self._train_metrics = train_metrics
        self._train_loader = train_loader
        self._opt = opt
        self._lr_scheduler = lr_scheduler
        self._max_epochs = max_epochs
        self._max_grad_norm = max_grad_norm

        self._test_metrics = test_metrics
        self._test_loader = test_loader
        self._epochs_per_test = epochs_per_test

        self._valid_loss = valid_loss
        self._valid_loader = valid_loader
        self._max_bad_valid_epochs = max_bad_valid_epochs
        self._best_valid_loss = float("inf")
        self._num_bad_valid_epochs = 0

        self._visualizer = visualizer

        self._writer = writer
        self._should_checkpoint_best_valid = should_checkpoint_best_valid

        ### Training

        self._trainer = Engine(self._train_batch)

        AverageMetric().attach(self._trainer)
        ProgressBar(persist=True).attach(self._trainer, ["loss"])

        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_training_info)

        ### Validation

        if early_stopping:
            self._validator = Engine(self._validate_batch)

            AverageMetric().attach(self._validator)
            ProgressBar(persist=False, desc="Validating").attach(self._validator)

            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._validate)

        ### Testing

        self._tester = Engine(self._test_batch)

        AverageMetric().attach(self._tester)
        ProgressBar(persist=False, desc="Testing").attach(self._tester)

        self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._test_and_log)

        ### Checkpointing

        if should_checkpoint_latest:
            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._save_checkpoint("latest"))

        try:
            self._load_checkpoint("latest")
        except FileNotFoundError:
            print("Did not find `latest' checkpoint.", file=sys.stderr)

            try:
                self._load_checkpoint("best_valid")
            except FileNotFoundError:
                print("Did not find `best_valid' checkpoint.", file=sys.stderr)

    def train(self):
        self._trainer.run(data=self._train_loader, max_epochs=self._max_epochs)

    def _train_batch(self, engine, batch):
        self._module.train()

        x, _ = batch # TODO: Potentially pass y also for genericity
        x = x.to(self._device)

        self._opt.zero_grad()

        train_metrics = self._train_metrics(self._module, x)
        loss = train_metrics["loss"]
        loss.backward()

        if self._max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._module.parameters(), self._max_grad_norm)

        self._opt.step()

        self._lr_scheduler.step()

        return train_metrics

    def test(self):
        self._module.eval()
        return self._tester.run(data=self._test_loader).metrics

    @torch.no_grad()
    def _test_and_log(self, engine):
        epoch = engine.state.epoch
        if (epoch - 1) % self._epochs_per_test == 0: # Test after first epoch
            for k, v in self.test().items():
                self._writer.write_scalar(f"test/{k}", v, global_step=engine.state.epoch)

                if not torch.isfinite(v):
                    self._save_checkpoint(tag="nan_during_test")

            self._visualizer.visualize(self._module, epoch)

    def _test_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        return self._test_metrics(self._module, x)

    @torch.no_grad()
    def _validate(self, engine):
        self._module.eval()

        state = self._validator.run(data=self._valid_loader)
        valid_loss = state.metrics["loss"]

        if valid_loss < self._best_valid_loss:
            print(f"Best validation loss {valid_loss} after epoch {engine.state.epoch}")
            self._num_bad_valid_epochs = 0
            self._best_valid_loss = valid_loss

            if self._should_checkpoint_best_valid:
                self._save_checkpoint(tag="best_valid")

        else:
            if not torch.isfinite(valid_loss):
                self._save_checkpoint(tag="nan_during_validation")

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
            for k, v in engine.state.output.items():
                self._writer.write_scalar("train/" + k, v, global_step=i)

        # TODO: Inefficient to recompute this if we are doing gradient clipping
        if i % self._STEPS_PER_GRAD_WRITE == 0:
            self._writer.write_scalar("train/grad-norm", self._get_grad_norm(), global_step=i)

        # TODO: We should do this _before_ calling self._lr_scheduler.step(), since
        # we will not correspond to the learning rate used at iteration i otherwise
        if i % self._STEPS_PER_LR_WRITE == 0:
            self._writer.write_scalar("train/lr", self._get_lr(), global_step=i)

    def _get_grad_norm(self):
        norm = 0
        for param in self._module.parameters():
            if param.grad is not None:
                norm += param.grad.norm().item()**2
        return np.sqrt(norm)

    def _get_lr(self):
        param_group, = self._opt.param_groups
        return param_group["lr"]

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

    def _load_checkpoint(self, tag):
        checkpoint = self._writer.load_checkpoint(tag, device=self._device)

        @self._trainer.on(Events.STARTED)
        def resume_trainer_state(engine):
            engine.state.epoch = checkpoint["epoch"]
            engine.state.iteration = checkpoint["iteration"]

        self._module.load_state_dict(checkpoint["module_state_dict"])
        self._opt.load_state_dict(checkpoint["opt_state_dict"])
        self._best_valid_loss = checkpoint["best_valid_loss"]
        self._num_bad_valid_epochs = checkpoint["num_bad_valid_epochs"]

        print(f"Loaded checkpoint `{tag}' after epoch {checkpoint['epoch']}", file=sys.stderr)
