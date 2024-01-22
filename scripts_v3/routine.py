import os
import numpy as np
from typing import Callable
from dataclasses import dataclass
from toolz import merge

import ssvepcca
from ssvepcca.utils import load_mat_data_array
from ssvepcca.algorithms import SSVEPAlgorithm


@dataclass
class ExperimentParams:
    """Class for keeping experiment parameters"""
    algorithm: SSVEPAlgorithm
    pipeline_function: Callable
    kwargs: dict
    name: str

    def __str__(self):
        return f"{self.algorithm.__name__}/{self.name}"


class TimeWindowParams:
    def __init__(
        self,
        start_time_idx,
        stop_time_idx,
        fit_start_time_idx,
        fit_stop_time_idx
    ):
        self.start_time_idx = start_time_idx
        self.stop_time_idx = stop_time_idx
        self.fit_start_time_idx = fit_start_time_idx
        self.fit_stop_time_idx = fit_stop_time_idx
    
    def __str__(self):
        return f"{self.start_time_idx}_{self.stop_time_idx}_{self.fit_start_time_idx}_{self.fit_stop_time_idx}"
    
    def get_params_dict(self):
        return {
            "start_time_idx": self.start_time_idx,
            "stop_time_idx": self.stop_time_idx,
            "fit_start_time_idx": self.fit_start_time_idx,
            "fit_stop_time_idx": self.fit_stop_time_idx,
        }


def run_experiment(
    experiment_params: ExperimentParams,
    time_window_params: TimeWindowParams,
    dataset_root_path: str,
    output_root_folder: str
):

    full_kwargs = merge(experiment_params.kwargs, time_window_params.get_params_dict())
    ssvep_algorithm = experiment_params.algorithm(**full_kwargs)

    output_experiment_folder = f"{output_root_folder}/{str(experiment_params)}/{str(time_window_params)}"
    print(f"    -Executing experiment: {output_experiment_folder}")

    for subject in range(1, ssvepcca.runtime_configuration.num_subjects + 1):
        print(f"        -Subject: S{subject}")

        data_path = f"{dataset_root_path}/S{subject}.mat"
        output_path = f"{output_experiment_folder}/S{subject}/"

        dataset = load_mat_data_array(data_path)
        predictions, predict_proba, accuracy = experiment_params.pipeline_function(dataset, ssvep_algorithm)

        os.makedirs(output_path, exist_ok=True)
        np.save(output_path + f"/predictions",      np.array(predictions),      allow_pickle=False)
        np.save(output_path + f"/predict_proba",    np.array(predict_proba),    allow_pickle=False)
        np.save(output_path + f"/accuracy",         np.array(accuracy),         allow_pickle=False)
