from time import process_time
import numpy as np

from ssvepcca import runtime_configuration as rc, parameters
from ssvepcca.utils import load_mat_data_array
from ssvepcca.pipelines import test_fit_predict, k_fold_predict
from ssvepcca.transformers import EEGType
from ssvepcca.algorithms import (
    StandardCCA, FilterbankCCA, SpatioTemporalCCA,
    StandardCCAFilter, FilterbankCCAFilter, SpatioTemporalCCAFilter,
    StandardCCAMulticlass, FilterbankCCAMulticlass, SpatioTemporalCCAMulticlass
)


if __name__ == "__main__":

    rc.load_from_name("tsinghua-bci-lab")

    DATASET_ROOT_PATH = "/media/cerqueirajc/windows_heavy_data/Ubuntu/masters/dataset/tsinghua_bci_lab"
    DATASET_PATH = DATASET_ROOT_PATH + "/S5.mat"

    input_data = load_mat_data_array(DATASET_PATH)

    start_time_index = 160
    stop_time_index = start_time_index + 125
    fit_start_time_idx = start_time_index
    fit_stop_time_idx = stop_time_index
    
    num_harmonics = 5
    electrode_list_fbcca = parameters.electrode_list_fbcca

    non_trainable_algos = [
        [
            "StandardCCA",
            StandardCCA(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                num_harmonics=num_harmonics,
            )
        ],
        [
            "FilterbankCCA",
            FilterbankCCA(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                num_harmonics=num_harmonics,
                fb_num_subband=10,
                fb_fundamental_freq=8,
                fb_upper_bound_freq=88,
            )
        ],
        [
            "SpatioTemporalCCA | SS",
            SpatioTemporalCCA(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                num_harmonics=num_harmonics,
                window_gap=2,
                window_length=1
            )
        ],
        [
            "SpatioTemporalCCA | FIR",
            SpatioTemporalCCA(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                num_harmonics=num_harmonics,
                window_gap=0,
                window_length=4
            ),
        ],
    ]

    trainable_algos = [
        [
            "StandardCCAFilter",
            StandardCCAFilter(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
            )
        ],
        [
            "FilterbankCCAFilter",
            FilterbankCCAFilter(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
                fb_num_subband=10,
                fb_fundamental_freq=8,
                fb_upper_bound_freq=88,
            )
        ],
        [
            "SpatioTemporalCCAFilter | SS",
            SpatioTemporalCCAFilter(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
                window_gap=2,
                window_length=1
            )
        ],
        [
            "SpatioTemporalCCAFilter | FIR",
            SpatioTemporalCCAFilter(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
                window_gap=0,
                window_length=4
            ),
        ],
        [
            "StandardCCAMulticlass",
            StandardCCAMulticlass(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
            )
        ],
        [
            "FilterbankCCAMulticlass",
            FilterbankCCAMulticlass(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
                fb_num_subband=10,
                fb_fundamental_freq=8,
                fb_upper_bound_freq=88,
            )
        ],
        [
            "SpatioTemporalCCAMulticlass | SS",
            SpatioTemporalCCAMulticlass(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
                window_gap=2,
                window_length=1
            )
        ],
        [
            "SpatioTemporalCCAMulticlass | FIR",
            SpatioTemporalCCAMulticlass(
                electrodes_name=electrode_list_fbcca,
                start_time_idx=start_time_index,
                stop_time_idx=stop_time_index,
                fit_start_time_idx=fit_start_time_idx,
                fit_stop_time_idx=fit_stop_time_idx,
                num_harmonics=num_harmonics,
                window_gap=0,
                window_length=4
            ),
        ],
    ]


    for name, algo in non_trainable_algos:
        pt_start = process_time()      
        res = test_fit_predict(input_data, algo)
        pt_stop = process_time()
        print(f"Name: {name}, accuracy: {res[2]}, process time: {pt_stop- pt_start}")

    for name, algo in trainable_algos:
        pt_start = process_time()      
        res = k_fold_predict(input_data, algo)
        pt_stop = process_time()
        print(f"Name: {name}, accuracy: {res[2]}, process time: {pt_stop- pt_start}")
