import scipy
import numpy as np

import ssvepcca

from configurations import experiments_correlation, experiments_filter
from routine import TimeWindowParams, run_experiment


if __name__ == "__main__":

    ssvepcca.runtime_configuration.load_from_name("tsinghua-beta-dataset-16-70")

    #DATASET_ROOT_PATH = "/media/cerqueirajc/windows_heavy_data/Ubuntu/masters/dataset/tsinghua_beta_dataset"
    DATASET_ROOT_PATH = "/mnt/mystorage/tsinghua_beta_dataset"
    #OUTPUT_ROOT_FOLDER = "results/tsinghua_beta_dataset"
    OUTPUT_ROOT_FOLDER = "/mnt/mystorage/results/tsinghua_beta_dataset" 

    time_window_end_values = [250, 375, 500, 625, 750, 875, 1000]

    # time_window_parameters = [
    #     TimeWindowParams(125, time_window_end, None, None)
    #     for time_window_end in time_window_end_values
    # ]

    time_window_parameters_full_data = [
        TimeWindowParams(125, time_window_end, 125, 1375)
        for time_window_end in time_window_end_values
    ]

    def load_mat_data_array_new(mat_path: str) -> np.ndarray:
        mat = scipy.io.loadmat(mat_path)
        return mat["data"][0][0][0].astype(float).T.transpose([1, 0, 2, 3])

    print(f"Running correlation algos:")
    for experiment_parameter in experiments_correlation:
        for time_window_params in time_window_parameters_full_data:
            run_experiment(
                experiment_parameter,
                time_window_params,
                DATASET_ROOT_PATH,
                OUTPUT_ROOT_FOLDER,
                load_mat_data_array_new
            )

    print(f"Running filter algos:")
    for experiment_parameter in experiments_filter:
        for time_window_params in time_window_parameters_full_data:
            run_experiment(
                experiment_parameter,
                time_window_params,
                DATASET_ROOT_PATH,
                OUTPUT_ROOT_FOLDER,
                load_mat_data_array_new
            )
