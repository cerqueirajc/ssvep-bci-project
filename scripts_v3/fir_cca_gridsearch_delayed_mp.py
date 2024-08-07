import os
import numpy as np
from multiprocessing import Pool

from ssvepcca import runtime_configuration as rc
from ssvepcca.utils import load_mat_data_array

from configurations import experiment_fir_gridsearch_correlation, experiment_fir_gridsearch_filter
from routine import TimeWindowParams, run_experiment


if __name__ == "__main__":
    
    rc.load_from_name("tsinghua-bci-lab")

    SCRIPT_NAME = "fir_cca_gridsearch_delayed_mp"
    DATASET_ROOT_PATH = "/mnt/mystorage/tsinghua_bci_lab"
    OUTPUT_ROOT_FOLDER = "/mnt/mystorage/results_delayed/tsinghua_bci_lab"

    stimulus_offset_seconds = 0.5
    visual_latency_seconds  = 0.14
    window_size_seconds = 2

    initial_time_value = (stimulus_offset_seconds + visual_latency_seconds) * rc.sample_frequency
    time_window_end_value = initial_time_value + window_size_seconds * rc.sample_frequency

    initial_stimulus_value = round(initial_time_value)
    final_stimulus_value = round(rc.num_samples - (stimulus_offset_seconds - visual_latency_seconds) * rc.sample_frequency)
    stimulus_window_end_value = round(time_window_end_value)

    time_window_default = TimeWindowParams(initial_stimulus_value, stimulus_window_end_value, None, None)
    time_window_extra_fit = TimeWindowParams(initial_stimulus_value, stimulus_window_end_value, initial_stimulus_value, final_stimulus_value)
    

    run_experiment_arglist = []
    
    print(f"Enqueue correlation algo")
    for experiment_parameter in experiment_fir_gridsearch_correlation:
        run_experiment_arglist.append((
            experiment_parameter,
            time_window_default,
            DATASET_ROOT_PATH,
            OUTPUT_ROOT_FOLDER,
            load_mat_data_array
        ))

    print(f"Enqueue supervised algo")
    for experiment_parameter in experiment_fir_gridsearch_filter:
        for time_params in [time_window_default, time_window_extra_fit]:
            run_experiment_arglist.append((
                experiment_parameter,
                time_params,
                DATASET_ROOT_PATH,
                OUTPUT_ROOT_FOLDER,
                load_mat_data_array
            ))

    os.makedirs(OUTPUT_ROOT_FOLDER, exist_ok=True)
    with open(f"{OUTPUT_ROOT_FOLDER}/{SCRIPT_NAME}.txt", "w") as f:
        for exp in run_experiment_arglist:
            f.write(str(exp[0]) + "/" + str(exp[1]) + '\n')

    with Pool(3) as p:
        p.starmap(run_experiment, run_experiment_arglist)
