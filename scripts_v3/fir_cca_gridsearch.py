import ssvepcca

from configurations import experiment_fir_gridsearch_correlation, experiment_fir_gridsearch_filter
from routine import TimeWindowParams, run_experiment


if __name__ == "__main__":
    
    ssvepcca.runtime_configuration.load_from_name("tsinghua-bci-lab")
    DATASET_ROOT_PATH = "/media/cerqueirajc/windows_heavy_data/Ubuntu/masters/dataset/tsinghua_bci_lab"
    OUTPUT_ROOT_FOLDER = "results/tsinghua_bci_lab"

    time_window_parameters = TimeWindowParams(125, 625, None, None)
    experiment_parameters_list = experiment_fir_gridsearch_correlation + experiment_fir_gridsearch_filter 
    
    print(f"Running fir_cca_hp_gridsearch script")
    for experiment_parameter in experiment_parameters_list:
        run_experiment(
            experiment_parameter,
            time_window_parameters,
            DATASET_ROOT_PATH,
            OUTPUT_ROOT_FOLDER
        )
