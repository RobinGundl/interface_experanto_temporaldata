from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)


class InterfaceExperiment:
    def __init__(
        self,
        cache_data: bool = True,
    ) -> None:
        
        """
        This Experiment class has no path to data, because td.Data objects are passed directly to the Interface.
        The data is cached in the interpolators. We put cache_data here for consistency, but it is not used.
        """
 
        self.devices = dict()
        self.start_time = np.inf
        self.end_time = -np.inf
        self.cache_data = cache_data
        self.meta = dict()

    @property
    def device_names(self):
        return tuple(self.devices.keys())

    def interpolate(self, times: slice, device=None) -> tuple[np.ndarray, np.ndarray]:
        if device is None:
            values = {}
            valid = {}
            for d, interp in self.devices.items():
                values[d], valid[d] = interp.interpolate(times)
        elif isinstance(device, str):
            assert device in self.devices, "Unknown device '{}'".format(device)
            values, valid = self.devices[device].interpolate(times)
        return values, valid

    def get_valid_range(self, device_name) -> tuple:
        return tuple(self.devices[device_name].valid_interval)
