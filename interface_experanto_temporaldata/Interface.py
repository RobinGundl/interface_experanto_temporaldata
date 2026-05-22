import os
import numpy as np

import temporaldata as td
from experanto.experiment import Experiment
from experanto.interpolators import ( 
    SequenceInterpolator, 
    TimeIntervalInterpolator, 
    ScreenInterpolator, 
    PhaseShiftedSequenceInterpolator, 
    SpikeInterpolator
)

from .InterfaceInterpolators import (
    InterfaceScreenInterpolator, 
    InterfaceSequenceInterpolator, 
    InterfaceTimeIntervalInterpolator, 
    InterfaceSpikeInterpolator
)
from .InterfaceExperiment import InterfaceExperiment


# custom function for conversion from temporaldata screen data to experanto screen data
# use it with interface_example.custom_fns = {(td.Data, "screen"): screen_fn}
def screen_fn(td_object, object_name):
    return InterfaceScreenInterpolator(data=td_object)

class Interface:
    """
    This class serves as an interface between experanto and temporaldata. 
    It can convert data from one format to the other,
    and it can also handle custom conversion functions.

    Attributes:
        datasets: a dictionary containing the datasets that can be processed by the interface
        custom_fns: a dictionary containing custom conversion functions,
                    where the keys are tuples of (type, name) and the values are the functions

    Example:
        >>> from InterfaceExperantoTemporaldata.Interface import Interface
        >>> interface = Interface(datasets={"dataset1": td_data, "dataset2": exp})
        >>> new_exp = interface.process_as_experanto("dataset1")
        >>> new_td_data = interface.process_as_temporaldata("dataset2")
    """
    def __init__(self, datasets, custom_fns: dict = None):
        if not isinstance(datasets, dict):
            raise TypeError("argument of the interface must be a dictionary")
        self.datasets = datasets
        self.custom_fns = custom_fns

    def process_as_experanto(self, dataset_name):
        """
        Converts given data (experanto.Experiment or temporaldata.Data) to an InterfaceExperiment

        Args:
            dataset_name: Name of the dataset within the dictionary of datasets

        Returns:
            An InterfaceExperiment object containing the data in the format of experanto
        """

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in the interface.")
        
        # getting the demanded dataset from the dictionary
        self.dataset = self.datasets[dataset_name]

        # nothing is being converted, as the experanto Experiment is already in the correct format
        if hasattr(self.dataset, "devices") and isinstance(self.dataset, Experiment):
            print("Exp -> Exp")

        # converting td.Data to an experanto Experiment
        elif isinstance(self.dataset, td.Data):
            print("Td -> Exp")
            data_object = self.dataset

            #creating an empty experiment object
            self.new_experiment = InterfaceExperiment()

            # calling the method for this specific conversion
            self._td_to_exp(data_object)
            self.dataset = self.new_experiment

        else:
            raise ValueError(f"Dataset {dataset_name} cannot be processed as Experanto.")
        
        return self.dataset


    def process_as_temporaldata(self, dataset_name):
        """
        Converts given data (experanto.Experiment or temporaldata.Data) to a td.Data object

        Args:
            dataset_name: Name of the dataset within the dictionary of datasets

        Returns:
            A td.Data object containing the data in the format of temporaldata
        """

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in the interface.")
        
        # getting the demanded dataset from the dictionary
        self.dataset = self.datasets[dataset_name]

        print("Processing dataset:", dataset_name)
        print("Dataset type:", type(self.dataset))

        # converting the experanto Experiment to td.Data
        if (hasattr(self.dataset, "devices") and isinstance(self.dataset, Experiment) or 
            isinstance(self.dataset, InterfaceExperiment)):
            print("Exp -> Td")

            # calling the method for this specific conversion
            self.dataset = self._exp_to_td(self.dataset)

        # nothing is being converted, as the td.Data object is already in the correct format
        elif isinstance(self.dataset, td.Data):
            print("Td -> Td")

        else:
            raise ValueError(f"Dataset {dataset_name} cannot be processed as Temporaldata.")

        return self.dataset

    def _td_to_exp(self, td_object, object_name=None):

        # checking if a custom function matches
        custom_fn_active = False
        if self.custom_fns is not None and len(self.custom_fns) > 0:
            for key in self.custom_fns.keys():
                if ((isinstance(td_object, key[0]) or key[0] is None) and
                    (object_name == key[1] or key[1] is None) and not
                    (key[0] is None and key[1] is None)):
                    fn = key
                    custom_fn_active = True

        if custom_fn_active:
            self.new_experiment.devices[object_name] = self.custom_fns[fn](td_object, object_name)

        # converting td.Data which is a container for other objects
        # the method operates recursively in order to convert all subdata
        elif isinstance(td_object, td.Data):
            for key in td_object.keys():
                if object_name is not None:
                    new_object_name = object_name + "." + key
                else:
                    new_object_name = key
                self._td_to_exp(getattr(td_object, key), new_object_name)

        elif isinstance(td_object, td.IrregularTimeSeries):
            self.new_experiment.devices[object_name] = InterfaceSpikeInterpolator(data=td_object)

        elif isinstance(td_object, td.RegularTimeSeries):
            self.new_experiment.devices[object_name] = InterfaceSequenceInterpolator(data=td_object)

        elif isinstance(td_object, td.Interval):
            for key in td_object.keys():
                # case where there is only start, end, and a third key
                # so we don't need the key name necessarily
                if len(td_object.keys()) == 3:
                    Interval_name = object_name
                else:
                    Interval_name = object_name + "." + key
                self.new_experiment.devices[Interval_name] = InterfaceTimeIntervalInterpolator(td_object, key)

        # saving ArrayDict object in the InterfaceExperiment meta
        elif isinstance(td_object, td.ArrayDict):
            data_dict = {key: getattr(td_object, key) for key in td_object.keys()}
            self.new_experiment.meta[object_name] = data_dict

        # saving data in the InterfaceExperiment meta
        elif isinstance(td_object, (int, float, str, bool, list, dict, np.bool_, np.integer, np.floating)):
            self.new_experiment.meta[object_name] = td_object

        else:
            raise TypeError(f"Unsupported Temporaldata type: {type(td_object)} for object '{object_name}'")

    def _exp_to_td(self, exp_object):

        # creating new Temporaldata.Data object
        new_td_data = td.Data(domain="auto")

        # going through all devices in the experanto.Experiment object
        for device_name, interpolator in exp_object.devices.items():

            # e.g. movement_phases.reach_period.test_mask ->
            # ["movement_phases", "reach_period", "test_mask"]
            attributes = device_name.split(".") 

            # the following if statement is only needed to revert the data, 
            # that already went through the default td->exp conversion
            if len(attributes) > 1:

                # e.g. ["movement_phases", "movement_phases.reach_period", 
                #       "movement_phases.reach_period.test_mask"]
                nested_attributes = _att_to_nested_att(attributes)

                # default case
                if not isinstance(interpolator, TimeIntervalInterpolator):
                    for na in nested_attributes[:-1]:
                        if not new_td_data.has_nested_attribute(na):
                            new_td_data.set_nested_attribute(na, td.Data(domain="auto"))

                # edge case for TimeIntervalInterpolators
                # (e.g. movement_phases.reach_period -> Interval instance
                #  with movement_phases.reach_period.test_mask an an attribute of this interval)
                elif (len(attributes) > 2 and
                    isinstance(interpolator, TimeIntervalInterpolator)):
                    for na in nested_attributes[:-2]:
                        if not new_td_data.has_nested_attribute(na):
                            new_td_data.set_nested_attribute(na, td.Data(domain="auto"))

            # checking if a custom function matches
            custom_fn_active = False
            if self.custom_fns is not None and len(self.custom_fns) > 0:
                for key in self.custom_fns.keys():
                    if ((isinstance(interpolator, key[0]) or key[0] is None) and 
                        (device_name == key[1] or key[1] is None) and not
                        (key[0] is None and key[1] is None)):
                        fn = key
                        custom_fn_active = True

            if custom_fn_active:
                merge_td_data(
                    new_td_data,
                    self.custom_fns[fn](interpolator, device_name)
                )

            elif isinstance(interpolator, SequenceInterpolator):
                if isinstance(interpolator._data, np.memmap):
                    data = np.array(interpolator._data).astype(np.float32)
                else:
                    data = interpolator._data

                # creating the RegularTimeSeries object and adding it to the td.Data object
                new_regular_timeseries = td.RegularTimeSeries(
                    sampling_rate=interpolator.sampling_rate,
                    data=data,
                    domain=td.Interval(start=interpolator.start_time, end=interpolator.end_time)
                )
                if isinstance(interpolator, PhaseShiftedSequenceInterpolator):
                    new_regular_timeseries._phase_shifts = interpolator._phase_shifts
                new_td_data.set_nested_attribute(device_name, new_regular_timeseries)

                # this if statement is only for InterfaceSequenceInterpolator instances
                if hasattr(interpolator, "meta"):
                    for attribute, value in interpolator.meta.items():
                        nested_attribute = device_name + "." + attribute
                        new_td_data.set_nested_attribute(nested_attribute, value)
                
                if (hasattr(interpolator, "root_folder") and
                    os.path.exists(interpolator.root_folder / "meta/unit_ids.npy")):
                    unit_ids = np.load(
                        interpolator.root_folder / "meta/unit_ids.npy"
                    )
                    units = td.ArrayDict(
                        id=unit_ids
                    )
                    new_td_data.units = units

            elif isinstance(interpolator, ScreenInterpolator):
                new_td_data.set_nested_attribute(device_name, td.Data(domain="auto"))

                # we create one IrregularTimeSeries for each trial, 
                # due to the different time dimensions
                for i, trial in enumerate(interpolator.trials):
                    data = trial.get_data()
                    if len(data.shape) == 2:
                        data = np.expand_dims(data, axis=0)
                    first_frame = interpolator._first_frame_idx[i]
                    n_frames = interpolator._num_frames[i]
                    timestamps = interpolator.timestamps[first_frame:first_frame + n_frames]

                    # creating the IrregularTimeSeries object and adding it to the td.Data object
                    new_irregular_timeseries = td.IrregularTimeSeries(
                        timestamps=timestamps,
                        data=data,
                        domain=td.Interval(interpolator.start_time, interpolator.end_time),
                        _metadata=trial._meta_data
                    )
                    new_td_data.set_nested_attribute(device_name + f".trial_{i}", new_irregular_timeseries)

            elif isinstance(interpolator, SpikeInterpolator):
                # e.g. indices: [0, 4, 6] -> 
                #      timestamps: [0, 0, 0, 0, 1, 1]
                if hasattr(interpolator, "indices"):
                    counts = np.diff(interpolator.indices)
                    unsorted_unit_index = np.concatenate([
                        np.full(count, i)
                        for i, count in enumerate(counts)
                    ])
                    order = np.lexsort((unsorted_unit_index, interpolator.spikes))
                    timestamps = interpolator.spikes[order]
                    unit_index = unsorted_unit_index[order]
                else:
                    timestamps = interpolator.spikes
                
                # creating the IrregularTimeSeries object and adding it to the td.Data object
                new_irregular_timeseries = td.IrregularTimeSeries(
                    timestamps=timestamps,
                    domain=td.Interval(interpolator.start_time, interpolator.end_time)
                )
                new_td_data.set_nested_attribute(device_name, new_irregular_timeseries)

                # this if statement is only for InterfaceSpikeInterpolator instances
                if hasattr(interpolator, "meta"):
                    for attribute, value in interpolator.meta.items():
                        nested_attribute = device_name + "." + attribute
                        new_td_data.set_nested_attribute(nested_attribute, value)
                        
                if hasattr(interpolator, "indices"):
                    new_td_data.set_nested_attribute(device_name + ".unit_index", unit_index)
                    new_td_data.units = td.ArrayDict(
                        id=np.unique(unit_index)
                    )

                if (hasattr(interpolator, "root_folder") and
                    os.path.exists(interpolator.root_folder / "meta/unit_ids.npy")):
                    unit_ids = np.load(
                        interpolator.root_folder / "meta/unit_ids.npy"
                    )
                    new_td_data.units = td.ArrayDict(
                        id=unit_ids
                    )

            elif isinstance(interpolator, TimeIntervalInterpolator):
                if not hasattr(interpolator, "labeled_intervals"):
                    try:
                        interpolator.labeled_intervals = {
                            label: np.load(interpolator.root_folder / filename)
                            for label, filename in interpolator.meta_labels.items()
                        }
                    except(FileNotFoundError, OSError) as e:
                        print(f"Error: File not found. Details: {e}")
                        
                # creating the Interval object and adding it to the td.Data object
                Interval_name = device_name.rsplit(".", 1)[0]

                # the following if statement is normally
                # for InterfaceTimeIntervalInterpolator instances
                # e.g. Interval_name = hold_period
                #      device_name = hold_period.test_mask
                if Interval_name != device_name:
                    # if the Interval does not exist yet, we have to create the start and end times
                    if not new_td_data.has_nested_attribute(Interval_name):
                        start, end = np.array([]), np.array([])
                        for key in interpolator.labeled_intervals.keys():
                            this_interval = interpolator.labeled_intervals[key]
                            for i in this_interval:
                                start = np.append(start, i[0])
                                end = np.append(end, i[-1])

                        new_td_data.set_nested_attribute(
                            Interval_name,
                            td.Interval(start=start, end=end)
                        )

                    # e.g. {True: np.array([[0, 1], [2, 3]]), False: np.array([[1, 2]])}
                    # -> {0: True, 1: False, 2: True}
                    attributes = {}
                    for key, values in interpolator.labeled_intervals.items():
                        for val in values:
                            attributes[val[0]] = key
                    attributes_sorted = {t: attributes[t] for t in sorted(attributes.keys())}

                    # {0: True, 1: False, 2: True} -> [True, False, True]
                    new_td_data.set_nested_attribute(device_name, np.array(list(attributes_sorted.values())))

                # default case
                else:
                    start, end = np.array([]), np.array([])
                    temp = {}
                    for key, values in interpolator.labeled_intervals.items():
                        for ti in values:
                            start = np.append(start, ti[0])
                            end = np.append(end, ti[-1])
                            temp[ti[0]] = key

                    temp_sorted = {t: temp[t] for t in sorted(temp.keys())}
                    Interval = td.Interval(
                        start=np.sort(start),
                        end=np.sort(end),
                        value=np.array(list(temp_sorted.values()))
                    )
                    new_td_data.set_nested_attribute(device_name, Interval)
            else:
                raise TypeError(f"Unsupported Interpolator type: {type(interpolator)} for device '{device_name}'")

        # only for the InterfaceExperiment
        if hasattr(exp_object, "meta"):
            # going through all meta data in the experanto.Experiment object (if existing)
            # and adding it to the td.Data object
            for meta_name, meta_value in exp_object.meta.items():
                attributes = meta_name.split(".")

                # creating the td.Data instance at first
                if len(attributes) > 1:
                    nested_attributes = _att_to_nested_att(attributes)
                    for n in nested_attributes[:-1]:
                        if not new_td_data.has_nested_attribute(n):
                            new_td_data.set_nested_attribute(n, td.Data(domain="auto"))     

                if isinstance(meta_value, dict):
                    new_td_data.set_nested_attribute(meta_name, td.ArrayDict())
                    for attribute, value in meta_value.items():
                        nested_attribute = meta_name + "." + attribute
                        new_td_data.set_nested_attribute(nested_attribute, value)

                else:
                    new_td_data.set_nested_attribute(meta_name, meta_value)

        return new_td_data


def _att_to_nested_att(attributes):
    # e.g. [a, b, c] -> [a, a.b, a.b.c]
    nested_attributes = []
    temp = ""
    for attr in attributes:
        if temp == "":
            temp = attr
        else:
            temp = temp + "." + attr
        nested_attributes.append(temp)
    return nested_attributes

def merge_td_data(object_1, object_2, attr_name=None):
    """
    Merges two td.Data instances into one td.Data.
    The first object will be containing the second object.
    """
    if isinstance(object_2, td.Data):
        for key in object_2.keys():
            if attr_name is not None:
                object_1.set_nested_attribute(attr_name, td.Data(domain="auto"))
                new_attr_name = attr_name + "." + key
            else:
                new_attr_name = key
            merge_td_data(object_1, getattr(object_2, key), new_attr_name)
    else:
        object_1.set_nested_attribute(attr_name, object_2)
