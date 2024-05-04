import ssvepcca
from ssvepcca.utils import load_mat_data_array

from configurations import experiments_correlation, experiments_filter, experiments_filter_short_training
from routine import TimeWindowParams, run_experiment


if __name__ == "__main__":

    ssvepcca.runtime_configuration.load_from_name("tsinghua-bci-lab")
    # DATASET_ROOT_PATH = "/media/cerqueirajc/windows_heavy_data/Ubuntu/masters/dataset/tsinghua_bci_lab"
    # OUTPUT_ROOT_FOLDER = "results/tsinghua_bci_lab"
    DATASET_ROOT_PATH = "/mnt/mystorage/tsinghua_bci_lab"
    OUTPUT_ROOT_FOLDER = "/mnt/mystorage/results_final/tsinghua_bci_lab" 

    time_window_end_values = [
        175,
        225,
        275,
        325,
        375,
        425,
        475,
        525,
        575,
        625,
        675,
        725,
        775,
        825,
        875,
        925,
        975,
        1025,
        1075,
        1125,
        1175,
        1225,
        1275,
        1325,
        1375,
    ]

    time_window_parameters = [
        TimeWindowParams(125, time_window_end, None, None)
        for time_window_end in time_window_end_values
    ]

    time_window_parameters_full_data = [
        TimeWindowParams(125, time_window_end, 125, 1375)
        for time_window_end in time_window_end_values
    ]

    print(f"Running correlation algos:")
    for experiment_parameter in experiments_correlation:
        for time_window_params in time_window_parameters:
            run_experiment(
                experiment_parameter,
                time_window_params,
                DATASET_ROOT_PATH,
                OUTPUT_ROOT_FOLDER,
                load_mat_data_array
            )

    print(f"Running filter algos:")
    for experiment_parameter in experiments_filter + experiments_filter_short_training:
        for time_window_params in time_window_parameters + time_window_parameters_full_data:
            run_experiment(
                experiment_parameter,
                time_window_params,
                DATASET_ROOT_PATH,
                OUTPUT_ROOT_FOLDER,
                load_mat_data_array
            )
