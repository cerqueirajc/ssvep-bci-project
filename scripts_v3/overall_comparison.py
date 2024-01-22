import ssvepcca

from configurations import experiments_correlation, experiments_filter
from routine import TimeWindowParams, run_experiment


if __name__ == "__main__":

    ssvepcca.runtime_configuration.load_from_name("tsinghua-bci-lab")
    DATASET_ROOT_PATH = "/media/cerqueirajc/windows_heavy_data/Ubuntu/masters/dataset/tsinghua_bci_lab"
    OUTPUT_ROOT_FOLDER = "results/tsinghua_bci_lab"

    time_window_end_values = [250, 375, 500, 625, 750, 875, 1000, 1125, 1250, 1375]

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
                OUTPUT_ROOT_FOLDER
            )

    print(f"Running filter algos:")
    for experiment_parameter in experiments_filter:
        for time_window_params in time_window_parameters + time_window_parameters_full_data:
            run_experiment(
                experiment_parameter,
                time_window_params,
                DATASET_ROOT_PATH,
                OUTPUT_ROOT_FOLDER
            )
