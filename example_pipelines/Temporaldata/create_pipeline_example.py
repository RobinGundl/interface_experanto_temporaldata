from pathlib import Path

import h5py
import numpy as np
import temporaldata as td

from experanto.experiment import Experiment
from experanto.configs import DEFAULT_MODALITY_CONFIG
from interface_experanto_temporaldata import Interface


BASE_DIR = Path(__file__).resolve().parent

ROOT_FOLDER = BASE_DIR / "experanto_000623"
OUTPUT = BASE_DIR / "data" / "processed" / "test" / "pipeline_example.h5"


def main():
    config = dict(DEFAULT_MODALITY_CONFIG)

    config["spikes"] = {
        "interpolation": {
            "_target_": "experanto.interpolators.SpikeInterpolator",
            "cache_data": True,
            "interpolation_window": 0.1,
            "interpolation_align": "center",
        }
    }

    for modality in ["lfp_macro", "lfp_micro", "eye_tracking", "pupil"]:
        config[modality] = {
            "interpolation": {
                "_target_": "experanto.interpolators.SequenceInterpolator"
            }
        }

    exp = Experiment(
        root_folder=ROOT_FOLDER,
        modality_config=config,
    )

    interface_dict = Interface({"example": exp})
    interface_example = interface_dict.process_as_temporaldata("example")

    interface_example.session = td.Data(id="pipeline_example")
    interface_example.subject = td.Data(id="c")
    interface_example.brainset = td.Data(id="test")

    interval = td.Interval(
        start=np.array([0.0]),
        end=np.array([1000.0]),
    )

    interface_example.test_domain = interval
    interface_example.valid_domain = interval
    interface_example.train_domain = interval

    interface_example.eye_tracking = interface_example.eye_tracking.to_irregular()
    interface_example.pupil = interface_example.pupil.to_irregular()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(OUTPUT, "w") as f:
        interface_example.to_hdf5(f)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()