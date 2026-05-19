from __future__ import annotations

import typing
import cv2
import numpy as np
import warnings
from scipy.ndimage import gaussian_filter1d

from experanto.interpolators import (
    SequenceInterpolator, 
    TimeIntervalInterpolator, 
    ScreenInterpolator, 
    PhaseShiftedSequenceInterpolator, 
    SpikeInterpolator
)
from experanto.intervals import TimeInterval
import temporaldata as td

class InterfaceSequenceInterpolator(SequenceInterpolator):
    def __init__(
        self,
        data: td.RegularTimeSeries,
        cache_data: bool = True,  # cached anyway, put it here for consistency
        keep_nans: bool = False,
        interpolation_mode: str = "nearest_neighbor",
        normalize: bool = False,
        normalize_subtract_mean: bool = False,
        normalize_std_threshold: typing.Optional[float] = None,  # or 0.01
        **kwargs,
    ) -> None:
        """
        interpolation_mode - nearest neighbor or linear
        keep_nans - if we keep nans in linear interpolation
        """

        self.meta = {}
        for key in data.keys():
            if isinstance(getattr(data, key), (int, float, str, bool, list, dict)):
                self.meta[key] = getattr(data, key)

        self.keep_nans = keep_nans
        self.interpolation_mode = interpolation_mode
        self.normalize = normalize
        self.normalize_subtract_mean = normalize_subtract_mean
        self.normalize_std_threshold = normalize_std_threshold
        self.sampling_rate = data.sampling_rate
        self.time_delta = 1.0 / self.sampling_rate
        self.start_time = data.domain.start[0]
        self.end_time = data.domain.end[-1]

        # Valid interval can be different to start time and end time.
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

        self._data = data.data
        self.n_signals = self._data.shape[1]

        if self.normalize and "mean" in self.meta and "stds" in self.meta:
            self.normalize_init()


    def normalize_init(self):
        self.mean = self.meta["mean"]
        self.std = self.meta["stds"]
        assert (
            self.mean.shape[0] == self.n_signals
        ), f"mean shape does not match: {self.mean.shape} vs {self._data.shape}"
        assert (
            self.std.shape[0] == self.n_signals
        ), f"std shape does not match: {self.std.shape} vs {self._data.shape}"
        self.mean = self.mean.T
        self.std = self.std.T
        if self.normalize_std_threshold:
            threshold = self.normalize_std_threshold * np.nanmean(self.std)
            idx = self.std > threshold
            self._precision = np.ones_like(self.std) / threshold
            self._precision[idx] = 1 / self.std[idx]
        else:
            self._precision = 1 / self.std

class InterfacePhaseShiftedSequenceInterpolator(PhaseShiftedSequenceInterpolator):
    def __init__(
        self,
        data: td.RegularTimeSeries,
        cache_data: bool = True,  # cached anyway, put it here for consistency
        keep_nans: bool = False,
        interpolation_mode: str = "nearest_neighbor",
        normalize: bool = False,
        normalize_subtract_mean: bool = False,
        normalize_std_threshold: typing.Optional[float] = None,  # or 0.01
        **kwargs,
    ) -> None:
        
        self.meta = {}
        for key in data.keys():
            if isinstance(getattr(data, key), (int, float, str, bool, list, dict)):
                self.meta[key] = getattr(data, key)

        self.keep_nans = keep_nans
        self.interpolation_mode = interpolation_mode
        self.normalize = normalize
        self.normalize_subtract_mean = normalize_subtract_mean
        self.normalize_std_threshold = normalize_std_threshold
        self.sampling_rate = data.sampling_rate
        self.time_delta = 1.0 / self.sampling_rate
        self.start_time = data.domain.start[0]
        self.end_time = data.domain.end[-1]

        self.n_signals = data._data.shape[1]
        self._data = data.data

        self._phase_shifts = data._phase_shifts
        if self.normalize and "mean" in self.meta and "stds" in self.meta:
            self.normalize_init()

        self.valid_interval = TimeInterval(
            self.start_time
            + (np.max(self._phase_shifts) if len(self._phase_shifts) > 0 else 0),
            self.end_time
            + (np.min(self._phase_shifts) if len(self._phase_shifts) > 0 else 0),
        )

    def normalize_init(self):
        self.mean = self.meta["mean"]
        self.std = self.meta["stds"]
        assert (
            self.mean.shape[0] == self.n_signals
        ), f"mean shape does not match: {self.mean.shape} vs {self._data.shape}"
        assert (
            self.std.shape[0] == self.n_signals
        ), f"std shape does not match: {self.std.shape} vs {self._data.shape}"
        self.mean = self.mean.T
        self.std = self.std.T
        if self.normalize_std_threshold:
            threshold = self.normalize_std_threshold * np.nanmean(self.std)
            idx = self.std > threshold
            self._precision = np.ones_like(self.std) / threshold
            self._precision[idx] = 1 / self.std[idx]
        else:
            self._precision = 1 / self.std

class InterfaceSpikeInterpolator(SpikeInterpolator):
    def __init__(
        self,
        data: td.IrregularTimeSeries,
        cache_data: bool = True,  # cached anyway, put it here for consistency
        interpolation_window: float = 0.3,
        interpolation_align: str = "center",
        smoothing_sigma: float = 0.0,
        interface_key: str | None = None, # tells us which attribute of the td.IrregularTimeSeries to use as the signal
    ):

        # save other data as meta information
        self.meta = {}
        for key in data.keys():
            if key not in ("timestamps", "start", "end"):
                self.meta[key] = getattr(data, key)

        self.start_time = data.domain.start[0]
        self.end_time = data.domain.end[-1]
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

        self.interpolation_window = interpolation_window
        self.interpolation_align = interpolation_align
        self.smoothing_sigma = smoothing_sigma

        self.has_indices = hasattr(data, "unit_index")
        self.interface_key = interface_key

        if self.has_indices:
            # conversion from 'unit_index' to 'indices',
            # which defines the start and end indices for each neuron's block
            # e.g. [1, 1, 2, 3] -> [0, 2, 3, 4]
            order = np.lexsort((data.timestamps, data.unit_index))
            self.spikes = data.timestamps[order]
            sorted_units = data.unit_index[order]

            _, first_idx = np.unique(sorted_units, return_index=True)
            self.indices = np.append(first_idx, len(sorted_units)).astype(np.int64)

            self.n_signals = len(self.indices) - 1
        else:
            self.spikes = data.timestamps

            if self.interface_key is None:
                self.n_signals = 2 # default to 2 signals (x and y position) if no interface key is provided
            else:
                if self.interface_key not in self.meta:
                    raise ValueError(
                        f"'{self.interface_key}' not found in meta. "
                        f"Available keys: {list(self.meta.keys())}"
                    )

                interface_data = np.asarray(self.meta[self.interface_key], dtype=np.float64)
                if interface_data.ndim == 1:
                    interface_data = interface_data[:, None]

                self.n_signals = interface_data.shape[1]

        if self.interpolation_align not in ["center", "left", "right"]:
            raise ValueError(
                f"Unknown alignment mode: {self.interpolation_align}, should be 'center', 'left' or 'right'"
            )

    def interpolate(
        self, times: np.ndarray, return_valid: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        # in this case, we just use the original interpolate function from SpikeInterpolator
        # (numba crashed, when I tried to use it, so I don't use 
        # return super().interpolate(times, return_valid=return_valid)
        # so I can rewrite _fast_count_spikes without numba)
        if self.has_indices:
            valid = self.valid_times(times)
            valid_times = times[valid]

            if len(valid_times) == 0:
                warnings.warn(
                    "Interpolation returns empty array, no valid times queried.",
                    UserWarning,
                    stacklevel=2,
                )
                return (
                    (np.empty((0, self.n_signals), dtype=np.float64), valid)
                    if return_valid
                    else np.empty((0, self.n_signals), dtype=np.float64)
                )

            if self.interpolation_align == "center":
                starts = valid_times - self.interpolation_window / 2
                ends = valid_times + self.interpolation_window / 2
            elif self.interpolation_align == "left":
                starts = valid_times
                ends = valid_times + self.interpolation_window
            elif self.interpolation_align == "right":
                starts = valid_times - self.interpolation_window
                ends = valid_times
            else:
                raise ValueError(
                    f"Unknown alignment mode: {self.interpolation_align}, should be 'center', 'left' or 'right'"
                )

            valid_size = len(valid_times)
            counts = np.zeros((valid_size, self.n_signals), dtype=np.float64)

            _count_spikes(self.spikes, self.indices, starts, ends, counts)

            if self.smoothing_sigma > 0:
                if valid_size > 1:
                    counts = gaussian_filter1d(counts, sigma=self.smoothing_sigma, axis=0)

            return (counts, valid) if return_valid else counts

        valid = self.valid_times(times)
        valid_times = times[valid]

        if len(valid_times) == 0:
            warnings.warn(
                "Interpolation returns empty array, no valid times queried.",
                UserWarning,
                stacklevel=2,
            )
            empty = np.empty((0, self.n_signals), dtype=np.float64)
            return (empty, valid) if return_valid else empty

        timestamps = np.asarray(self.spikes, dtype=np.float64)
        data = np.asarray(self.meta[self.interface_key], dtype=np.float64)

        if data.ndim == 1:
            data = data[:, None]

        idx_right = np.searchsorted(timestamps, valid_times, side="left")
        idx_right = np.clip(idx_right, 1, len(timestamps) - 1)
        idx_left = idx_right - 1

        t0 = timestamps[idx_left]
        t1 = timestamps[idx_right]
        y0 = data[idx_left]
        y1 = data[idx_right]

        denom = t1 - t0
        denom[denom == 0] = 1.0

        w = ((valid_times - t0) / denom)[:, None]
        out = (1.0 - w) * y0 + w * y1

        return (out, valid) if return_valid else out

class InterfaceTimeIntervalInterpolator(TimeIntervalInterpolator):
    def __init__(self, data: td.Interval, key: str, **kwargs):
        self.cache_data = True

        self.start_time = data.start[0]
        self.end_time = data.end[-1]

        self.valid_interval = TimeInterval(self.start_time, self.end_time)
        self.labeled_intervals = {}

        # going through the values of the corresponding attribute of the td.Interval
        # and creating the labeled intervals
        for i, k in enumerate(getattr(data, key)):
            k_hashable = ensure_hashable(k)
            self.labeled_intervals.setdefault(k_hashable, []).append(
                [data.start[i], data.end[i]]
            )

        # sorting
        for k_hashable, intervals in self.labeled_intervals.items():
            temp = np.array(intervals)
            self.labeled_intervals[k_hashable] = temp[temp[:, 0].argsort()]

def ensure_hashable(val):
    try:
        hash(val)
        return val
    except TypeError:
        return tuple(val)
    
class InterfaceScreenInterpolator(ScreenInterpolator):
    def __init__(
        self,
        data : td.Data,
        cache_data: bool = True,  # cached anyway, put it here for consistency      
        rescale: bool = False,
        rescale_size: tuple[int, int] | None = None,
        normalize: bool = False,
        **kwargs,
    ) -> None:
        """
        rescale would rescale images to the _image_size if true
        cache_data: if True, loads and keeps all trial data in memory
        """
    
        timestamps = []
        self.trials = []
        for key in data.keys():
            trial = getattr(data, key)

            # getting/creating the metadata
            if hasattr(trial, "_metadata"):
                metadata = trial._metadata
            else:
                if len(trial.data.shape) == 2:
                    modality = "blank"
                elif trial.data.shape[0] > 1:
                    modality = "video"
                else:
                    modality = "image"
                    
                metadata = {
                    "first_frame_idx": len(timestamps),
                    "image_size": [trial.data.shape[-2], trial.data.shape[-1]],
                    "modality": modality,
                    "num_frames": len(trial.timestamps)
                }

            try:
                timestamps = timestamps + trial.timestamps
            except ValueError:
                timestamps = np.append(timestamps, trial.timestamps)

            self.trials.append(
                ScreenTrial.create(
                    key, metadata, cached_data=trial.data
                )
            )

        self.timestamps = timestamps
        self.start_time = data.domain.start
        self.end_time = data.domain.end
        self.valid_interval = TimeInterval(self.start_time, self.end_time)
        self.rescale = rescale

        # create mapping from image index to file index
        self._num_frames = [t.num_frames for t in self.trials]
        self._first_frame_idx = [t.first_frame_idx for t in self.trials]
        self._data_file_idx = np.concatenate(
            [np.full(t.num_frames, i) for i, t in enumerate(self.trials)]
        )

        # infer image size
        if not rescale_size:
            for m in self.trials:
                if m.image_size is not None:
                    self._image_size = m.image_size
                    break
        else:
            self._image_size = rescale_size
        self.normalize = normalize
        if self.normalize:
            self.normalize_init()

    def normalize_init(self):
        self.mean = np.load(self.root_folder / "meta/means.npy")
        self.std = np.load(self.root_folder / "meta/stds.npy")
        if self.rescale:
            self.mean = self.rescale_frame(self.mean.T).T
            self.std = self.rescale_frame(self.std.T).T
        assert (
            self.mean.shape == self._image_size
        ), f"mean size is different: {self.mean.shape} vs {self._image_size}"
        assert (
            self.std.shape == self._image_size
        ), f"std size is different: {self.std.shape} vs {self._image_size}"

    def normalize_data(self, data):
        return (data - self.mean) / self.std

    def rescale_frame(self, frame: np.array) -> np.array:
        """
        Changes the resolution of the image to this size.
        Returns: Rescaled image
        """
        return cv2.resize(frame, self._image_size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )

class ScreenTrial:
    def __init__(
        self,
        data_file_name: str,
        meta_data: dict,
        image_size: tuple,
        first_frame_idx: int,
        num_frames: int,
        cached_data: np.ndarray,
    ) -> None:
        self.data_file_name = data_file_name
        self._meta_data = meta_data
        self.modality = meta_data.get("modality")
        self.image_size = image_size
        self.first_frame_idx = first_frame_idx
        self.num_frames = num_frames
        self._cached_data = cached_data

    @staticmethod
    def create(
        data_file_name: str, meta_data: dict, cached_data: np.ndarray
    ) -> "ScreenTrial":
        modality = meta_data.get("modality")
        class_name = modality.lower().capitalize() + "Trial"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](data_file_name, meta_data, cached_data=cached_data)

    def get_data(self) -> np.array:
        return self._cached_data

    def get_meta(self, property: str):
        return self._meta_data.get(property)


class ImageTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cached_data) -> None:
        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
            cached_data=cached_data,
        )


class VideoTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cached_data) -> None:
        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            meta_data.get("num_frames"),
            cached_data=cached_data,
        )


class BlankTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cached_data) -> None:
        self.interleave_value = meta_data.get("interleave_value")

        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
            cached_data=cached_data,
        )

    def get_data_(self) -> np.array:
        """Override base implementation to generate blank data"""
        return np.full((1,) + self.image_size, self.interleave_value, dtype=np.float32)


class InvalidTrial(ScreenTrial):
    def __init__(self, data_file_name, meta_data, cached_data) -> None:
        self.interleave_value = meta_data.get("interleave_value")

        super().__init__(
            data_file_name,
            meta_data,
            tuple(meta_data.get("image_size")),
            meta_data.get("first_frame_idx"),
            1,
            cached_data=cached_data,
        )

    def get_data_(self) -> np.array:
        """Override base implementation to generate blank data"""
        return np.full((1,) + self.image_size, self.interleave_value, dtype=np.float32)


def _lower_bound(arr, left, right, x):
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left

# _fast_count_spikes without numba
def _count_spikes(all_spikes, indices, window_starts, window_ends, out_counts):
    n_batch = len(window_starts)
    n_neurons = len(indices) - 1

    for i in range(n_neurons):
        idx_start = indices[i]
        idx_end = indices[i + 1]

        for b in range(n_batch):
            t0 = window_starts[b]
            t1 = window_ends[b]

            c_start = _lower_bound(all_spikes, idx_start, idx_end, t0)
            c_end = _lower_bound(all_spikes, idx_start, idx_end, t1)

            out_counts[b, i] = c_end - c_start
