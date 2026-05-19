import torchmetrics
import torch_brain

from torch_brain.data import Dataset
from torch_brain.registry import register_modality, DataType


# Neue Modality: pupil (1D)
register_modality(
    "pupil_1d",
    dim=1,
    type=DataType.CONTINUOUS,
    timestamp_key="pupil.timestamps",
    value_key="pupil.data",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)


class TransformContainer:
    def __init__(self, transforms=None):
        self.transforms = list(transforms) if transforms is not None else []

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class MyDataset(Dataset):
    RECORDING_ID = "test/pipeline_example"

    READOUT_CONFIG = {
        "readout": {
            "readout_id": "pupil_1d",
            "normalize_mean": 0.0,
            "normalize_std": 1.0,
            "metrics": [],   # erstmal aus für Stabilität
        }
    }

    def __init__(self, root, transform=None):
        super().__init__(
            root=root,
            recording_id=self.RECORDING_ID,
            transform=None,
        )

        self.recording_ids = [self.RECORDING_ID]

        if transform is None:
            self.transform = TransformContainer()
        elif hasattr(transform, "transforms"):
            self.transform = transform
        else:
            self.transform = TransformContainer([transform])

    def _attach_config(self, data):
        data.config = self.READOUT_CONFIG
        return data

    def get_recording(self, recording_id):
        if recording_id != self.RECORDING_ID:
            raise KeyError(f"Unknown recording_id: {recording_id}")

        data = self.get_recording_data(recording_id)
        return self._attach_config(data)

    def get_sampling_intervals(self, split):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"Unknown split: {split}")

        recording = self.get_recording(self.RECORDING_ID)
        interval = getattr(recording, f"{split}_domain")
        return {self.RECORDING_ID: interval}

    def __getitem__(self, idx):
        original_transform = self.transform
        self.transform = None
        try:
            sample = super().__getitem__(idx)
        finally:
            self.transform = original_transform

        sample = self._attach_config(sample)
        sample = self.transform(sample)
        return sample