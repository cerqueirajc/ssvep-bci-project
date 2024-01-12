import os
import numpy as np
from toolz import merge

import ssvepcca
from ssvepcca.utils import load_mat_data_array
from configurations import experiment_fir_gridsearch_correlation, experiment_fir_gridsearch_filter


def save_results(output_folder, results):
    os.makedirs(output_folder, exist_ok=True)
    for name, result_array in results.items():
        np.save(output_folder + f"/{name}", np.array(result_array), allow_pickle=False)


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


if __name__ == "__main__":
    
    ssvepcca.runtime_configuration.load_from_name("tsinghua-bci-lab")
    DATASET_ROOT_PATH = "/media/cerqueirajc/windows_heavy_data/Ubuntu/masters/dataset/tsinghua_bci_lab"
    OUTPUT_ROOT_FOLDER = "results/tsinghua_bci_lab"

    time_window_parameter = TimeWindowParams(125, 625, None, None)

    print(f"Running fir_cca_hp_gridsearch script")

    for experiment_params in experiment_fir_gridsearch_correlation + experiment_fir_gridsearch_filter:

        full_kwargs = merge(experiment_params.kwargs, time_window_parameter.get_params_dict())
        ssvep_algorithm = experiment_params.algorithm(**full_kwargs)

        output_experiment_folder = f"{OUTPUT_ROOT_FOLDER}/{str(experiment_params)}/{str(time_window_parameter)}"
        print(f"    -Executing experiment: {output_experiment_folder}")

        for subject in range(1, ssvepcca.runtime_configuration.num_subjects + 1):
            print(f"        -Subject: S{subject}")

            data_path = f"{DATASET_ROOT_PATH}/S{subject}.mat"
            output_path = f"{output_experiment_folder}/S{subject}/"

            dataset = load_mat_data_array(data_path)
            predictions, predict_proba, accuracy = experiment_params.pipeline_function(dataset, ssvep_algorithm)

            os.makedirs(output_path, exist_ok=True)
            np.save(output_path + f"/predictions", np.array(predictions), allow_pickle=False)
            np.save(output_path + f"/predict_proba", np.array(predict_proba), allow_pickle=False)
            np.save(output_path + f"/accuracy", np.array(accuracy), allow_pickle=False)
