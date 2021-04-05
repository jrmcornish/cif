import os
import datetime
import json
import sys

import torch

from tensorboardX import SummaryWriter


class Tee:
    def __init__(self, primary_file, secondary_file):
        self.primary_file = primary_file
        self.secondary_file = secondary_file

        self.encoding = self.primary_file.encoding

    # TODO: Should redirect all attrs to primary_file if not found here.
    def isatty(self):
        return self.primary_file.isatty()

    def fileno(self):
        return self.primary_file.fileno()

    def write(self, data):
        # We get problems with ipdb if we don't do this:
        if isinstance(data, bytes):
            data = data.decode()

        self.primary_file.write(data)
        self.secondary_file.write(data)

    def flush(self):
        self.primary_file.flush()
        self.secondary_file.flush()


# TODO: Rename to LogDir or something since we load as well as write. Could make
# a base class that knows all directories, and then inherit ReadOnlyLogDir and
# WriteableLogDir.
class Writer:
    _STDOUT = sys.stdout
    _STDERR = sys.stderr

    def __init__(self, logdir, make_subdir, tag_group):
        if make_subdir:
            os.makedirs(logdir, exist_ok=True)

            timestamp = f"{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
            logdir = os.path.join(logdir, timestamp)

        self._writer = SummaryWriter(logdir=logdir)

        assert logdir == self._writer.logdir
        self._logdir = logdir

        self._tag_group = tag_group

        LINE_BUFFERING = 1

        sys.stdout = Tee(
            primary_file=self._STDOUT,
            secondary_file=open(os.path.join(logdir, "stdout"), "a", buffering=LINE_BUFFERING)
        )

        sys.stderr = Tee(
            primary_file=self._STDERR,
            secondary_file=open(os.path.join(logdir, "stderr"), "a", buffering=LINE_BUFFERING)
        )

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

    def write_textfile(self, tag, text):
        path = os.path.join(self._logdir, f"{tag}.txt")
        with open(path, "w") as f:
            f.write(text)

    def write_checkpoint(self, tag, data):
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        checkpoint_path = self._checkpoint_path(tag)

        tmp_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"{os.path.basename(checkpoint_path)}.tmp"
        )

        torch.save(data, tmp_checkpoint_path)
        # replace is atomic, so we guarantee our checkpoints are always good
        os.replace(tmp_checkpoint_path, checkpoint_path)

    def load_checkpoint(self, tag, device):
        return torch.load(self._checkpoint_path(tag), map_location=device)

    def _checkpoint_path(self, tag):
        return os.path.join(self._checkpoints_dir, f"{tag}.pt")

    @property
    def _checkpoints_dir(self):
        return os.path.join(self._logdir, "checkpoints")

    def _tag(self, tag):
        return f"{self._tag_group}/{tag}"


# XXX: A bit ugly - we only inherit to ensure we have _checkpoint_path()
class DummyWriter(Writer):
    def __init__(self, logdir):
        self._logdir = logdir

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

    def write_textfile(self, tag, text):
        pass

    def write_checkpoint(self, tag, data):
        pass

    def load_checkpoint(self, tag, device):
        if self._logdir is None:
            raise FileNotFoundError
        else:
            return super().load_checkpoint(tag, device)
