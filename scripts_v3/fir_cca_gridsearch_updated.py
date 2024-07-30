import os
from multiprocessing import Pool

import ssvepcca
from ssvepcca.utils import load_mat_data_array

from configurations import experiment_fir_gridsearch_correlation, experiment_fir_gridsearch_filter
from routine import TimeWindowParams, run_experiment


if __name__ == "__main__":

    ssvepcca.runtime_configuration.load_from_name("tsinghua-bci-lab")
    DATASET_ROOT_PATH = "/mnt/mystorage/tsinghua_bci_lab"
    OUTPUT_ROOT_FOLDER = "/mnt/mystorage/results_fir_opt/tsinghua_bci_lab"
    SCRIPT_NAME = "fir_cca_gridsearch_updated"

    # 125_250_125_1375
    time_window_parameters = TimeWindowParams(125, 625, 125, 1375)
    # experiment_parameters_list = experiment_fir_gridsearch_correlation + experiment_fir_gridsearch_filter
    experiment_parameters_list = experiment_fir_gridsearch_filter

    run_experiment_arglist = []

    print(f"Running fir_cca_hp_gridsearch script")
    for experiment_parameter in experiment_parameters_list:
        run_experiment_arglist.append((
            experiment_parameter,
            time_window_parameters,
            DATASET_ROOT_PATH,
            OUTPUT_ROOT_FOLDER,
            load_mat_data_array
        ))

    os.makedirs(OUTPUT_ROOT_FOLDER, exist_ok=True)
    with open(f"{OUTPUT_ROOT_FOLDER}/{SCRIPT_NAME}.txt", "w") as f:
        for exp in run_experiment_arglist:
            f.write(str(exp[0]) + "/" + str(exp[1]) + '\n')

    with Pool(8) as p:
        p.starmap(run_experiment, run_experiment_arglist)
