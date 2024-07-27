import os
import numpy as np
from multiprocessing import Pool


from ssvepcca import runtime_configuration as rc
from ssvepcca.utils import load_mat_data_array

from configurations import experiments_correlation, experiments_filter, experiments_new_cca_multiclass
from routine import TimeWindowParams, run_experiment


if __name__ == "__main__":

    rc.load_from_name("tsinghua-bci-lab")

    SCRIPT_NAME = "exp_benchmark_dataset_delayed_mp"
    DATASET_ROOT_PATH = "/mnt/mystorage/tsinghua_bci_lab"
    OUTPUT_ROOT_FOLDER = "/mnt/mystorage/results_delayed/tsinghua_bci_lab" 

    stimulus_offset_seconds = 0.5
    visual_latency_seconds  = 0.14
    step_size_seconds       = 0.2

    initial_time_value = (stimulus_offset_seconds + visual_latency_seconds) * rc.sample_frequency
    step_size_value = 0.2 * rc.sample_frequency
    
    time_window_end_values = np.round(np.arange(
        initial_time_value + step_size_value,
        rc.num_samples,
        step_size_value
    )).astype(int).tolist()

    initial_stimulus_value = round(initial_time_value)
    final_stimulus_value = round(rc.num_samples - (stimulus_offset_seconds - visual_latency_seconds) * rc.sample_frequency)


    time_window_parameters = [
        TimeWindowParams(initial_stimulus_value, time_window_end, None, None)
        for time_window_end in time_window_end_values
    ]

    time_window_parameters_full_data = [
        TimeWindowParams(initial_stimulus_value, time_window_end, initial_stimulus_value, final_stimulus_value)
        for time_window_end in time_window_end_values
    ]

    run_experiment_arglist = []

    print(f"Enqueue correlation algos:")
    for experiment_parameter in experiments_correlation:
        for time_window_params in time_window_parameters:
            run_experiment_arglist.append((
                experiment_parameter,
                time_window_params,
                DATASET_ROOT_PATH,
                OUTPUT_ROOT_FOLDER,
                load_mat_data_array
            ))

    print(f"Enqueue filter algos:")
    for experiment_parameter in experiments_filter + experiments_new_cca_multiclass:
        for time_window_params in time_window_parameters + time_window_parameters_full_data:
            run_experiment_arglist.append((
                experiment_parameter,
                time_window_params,
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
