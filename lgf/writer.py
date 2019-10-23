import os
import datetime
import json

import torch

from tensorboardX import SummaryWriter


class Writer:
    def __init__(self, logdir_root, tag_group):
        os.makedirs(logdir_root, exist_ok=True)

        timestamp = f"{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
        logdir = os.path.join(logdir_root, timestamp)

        self._writer = SummaryWriter(logdir=logdir)

        self._tag_group = tag_group

    def write_scalar(self, tag, scalar_value, global_step=None):
        self._writer.add_scalar(self._tag(tag), scalar_value, global_step=global_step)

    def write_image(self, tag, img_tensor, global_step=None):
        self._writer.add_image(self._tag(tag), img_tensor, global_step=global_step)

    def write_figure(self, tag, figure, global_step=None):
        self._writer.add_figure(self._tag(tag), figure, global_step=global_step)

    def write_hparams(self, hparam_dict=None, metric_dict=None):
        self._writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    def write_json(self, tag, data):
        text = json.dumps(data, indent=4)

        self._writer.add_text(
            self._tag(tag),
            4*" " + text.replace("\n", "\n" + 4*" ") # Indent by 4 to ensure codeblock formatting
        )

        json_path = os.path.join(self._logdir, f"{tag}.json")

        with open(json_path, "w") as f:
            f.write(text)

    def write_checkpoint(self, tag, data):
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(self._checkpoints_dir, f"{tag}.pt")

        tmp_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"{os.path.basename(checkpoint_path)}.tmp"
        )

        torch.save(data, tmp_checkpoint_path)
        # rename is atomic, so we guarantee our checkpoints are always good
        os.rename(tmp_checkpoint_path, checkpoint_path)

    @property
    def _checkpoints_dir(self):
        return os.path.join(self._logdir, "checkpoints")

    @property
    def _logdir(self):
        return self._writer.logdir

    def _tag(self, tag):
        return f"{self._tag_group}/{tag}"


class DummyWriter:
    def write_scalar(self, tag, scalar_value, global_step=None):
        pass

    def write_image(self, tag, img_tensor, global_step=None):
        pass

    def write_figure(self, tag, figure, global_step=None):
        pass

    def write_hparams(self, hparam_dict=None, metric_dict=None):
        pass

    def write_json(self, tag, data):
        pass

    def write_checkpoint(self, tag, data):
        pass
