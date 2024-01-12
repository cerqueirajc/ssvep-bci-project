from typing import Callable, Tuple
from toolz import merge
from dataclasses import dataclass
import numpy as np

from ssvepcca import parameters
from ssvepcca.pipelines import test_fit_predict, k_fold_predict
from ssvepcca.algorithms import (
    SSVEPAlgorithm, StandardCCA, FilterbankCCA,
    SpatioTemporalCCA, FBSpatioTemporalCCA, StandardCCAFilter,
    FilterbankCCAFilter, SpatioTemporalCCAFilter,
    FBSpatioTemporalCCAFilter
)


PARAMS_CCA_SINGLE_COMPONENT = {
    "electrodes_name": parameters.electrode_list_fbcca,
    "num_harmonics": 5,
}

PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT = merge(
    PARAMS_CCA_SINGLE_COMPONENT,
    {
        "window_gap": 3,
        "window_length": 1,
    }
)

PARAMS_FIR_CCA_DEFAULT = merge(
    PARAMS_CCA_SINGLE_COMPONENT,
    {
        "window_gap": 0,
        "window_length": 5,
    }
)

PARAMS_FILTERBANK_CCA = merge(
    PARAMS_CCA_SINGLE_COMPONENT,
    {
        "fb_num_subband": 10,
        "fb_fundamental_freq": 8,
        "fb_upper_bound_freq": 88,
        "fb_weight__a": 1.25,
        "fb_weight__b": 0.25,
    }
)

PARAMS_FB_SS_CCA_DEFAULT = merge(
    PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT,
    PARAMS_FILTERBANK_CCA
)

PARAMS_FB_FIR_CCA_DEFAULT = merge(
    PARAMS_FIR_CCA_DEFAULT,
    PARAMS_FILTERBANK_CCA
)

def ss_fir_params_str(window_gap, window_length):
    return f"G{window_gap}__L{window_length}"


@dataclass
class ExperimentParams:
    """Class for keeping experiment parameters"""
    algorithm: SSVEPAlgorithm
    pipeline_function: Callable
    kwargs: dict
    name: str

    def __str__(self):
        return f"{self.algorithm.__name__}/{self.name}"


experiments_correlation: Tuple[ExperimentParams, ...] = (
    ExperimentParams(StandardCCA, test_fit_predict, PARAMS_CCA_SINGLE_COMPONENT, "default"),
    ExperimentParams(SpatioTemporalCCA, test_fit_predict, PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT, "default"),
    ExperimentParams(SpatioTemporalCCA, test_fit_predict, PARAMS_FIR_CCA_DEFAULT, "default"),
    ExperimentParams(FilterbankCCA, test_fit_predict, PARAMS_FILTERBANK_CCA, "default"),
    ExperimentParams(FBSpatioTemporalCCA, test_fit_predict, PARAMS_FB_SS_CCA_DEFAULT, "default"),
)


experiments_filter: Tuple[ExperimentParams, ...] = (
    ExperimentParams(StandardCCAFilter, k_fold_predict, PARAMS_CCA_SINGLE_COMPONENT, "default"),
    ExperimentParams(SpatioTemporalCCAFilter, k_fold_predict, PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT, "default"),
    ExperimentParams(SpatioTemporalCCAFilter, k_fold_predict, PARAMS_FIR_CCA_DEFAULT, "default"),
    ExperimentParams(FilterbankCCAFilter, k_fold_predict, PARAMS_FILTERBANK_CCA, "default"),
    ExperimentParams(FBSpatioTemporalCCAFilter, k_fold_predict, PARAMS_FB_SS_CCA_DEFAULT, "default")
)


SS_CCA_WINDOW_GAP_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
SS_WINDOW_LENGTH = 1

experiment_ss_hp_tuning_correlation: Tuple[ExperimentParams, ...] = tuple(
    ExperimentParams(
        SpatioTemporalCCA,
        test_fit_predict,
        merge(
            PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT,
            {"window_gap": window_gap, "window_length": SS_WINDOW_LENGTH}
        ),
        ss_fir_params_str(window_gap, SS_WINDOW_LENGTH)
    )
    for window_gap in SS_CCA_WINDOW_GAP_LIST
)

experiment_ss_hp_tuning_filter: Tuple[ExperimentParams, ...] = tuple(
    ExperimentParams(
        SpatioTemporalCCAFilter,
        k_fold_predict,
        merge(
            PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT,
            {"window_gap": window_gap, "window_length": SS_WINDOW_LENGTH}
        ),
        ss_fir_params_str(window_gap, SS_WINDOW_LENGTH)
    )
    for window_gap in SS_CCA_WINDOW_GAP_LIST
)   


FIR_CCA_WINDOW_GAP = 0
FIR_CCA_WINDOW_LENGTH_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]

experiment_fir_hp_tuning_correlation: Tuple[ExperimentParams, ...] = tuple(
    ExperimentParams(
        SpatioTemporalCCA,
        test_fit_predict,
        merge(
            PARAMS_FIR_CCA_DEFAULT,
            {"window_gap": FIR_CCA_WINDOW_GAP, "window_length": window_length}
        ),
        ss_fir_params_str(FIR_CCA_WINDOW_GAP, window_length)
    )
    for window_length in FIR_CCA_WINDOW_LENGTH_LIST
)

experiment_fir_hp_tuning_filter: Tuple[ExperimentParams, ...] = tuple(
    ExperimentParams(
        SpatioTemporalCCAFilter,
        k_fold_predict,
        merge(
            PARAMS_FIR_CCA_DEFAULT,
            {"window_gap": FIR_CCA_WINDOW_GAP, "window_length": window_length}
        ),
        ss_fir_params_str(FIR_CCA_WINDOW_GAP, window_length)
    )
    for window_length in FIR_CCA_WINDOW_LENGTH_LIST
)


FIR_GRIDSEARCH_GAP    = [1, 2, 3, 4, 5]
FIR_GRIDSEARCH_LENGTH = [2, 3, 4, 5, 6, 7, 8, 9, 10]

experiment_fir_gridsearch_correlation: Tuple[ExperimentParams, ...] = tuple(
    ExperimentParams(
        SpatioTemporalCCA,
        test_fit_predict,
        merge(
            PARAMS_FIR_CCA_DEFAULT,
            {"window_gap": window_gap, "window_length": window_length}
        ),
        ss_fir_params_str(window_gap, window_length)
    )
    for window_gap in FIR_GRIDSEARCH_GAP
    for window_length in FIR_GRIDSEARCH_LENGTH
)

experiment_fir_gridsearch_filter: Tuple[ExperimentParams, ...] = tuple(
    ExperimentParams(
        SpatioTemporalCCAFilter,
        k_fold_predict,
        merge(
            PARAMS_FIR_CCA_DEFAULT,
            {"window_gap": window_gap, "window_length": window_length}
        ),
        ss_fir_params_str(window_gap, window_length)
    )
    for window_gap in FIR_GRIDSEARCH_GAP
    for window_length in FIR_GRIDSEARCH_LENGTH
)




if __name__ == "__main__":
    for rparams in (
        experiments_correlation
        + experiments_filter
        + experiment_ss_hp_tuning_correlation
        + experiment_ss_hp_tuning_filter
        + experiment_fir_hp_tuning_correlation
        + experiment_fir_hp_tuning_filter
    ):
        print(rparams, rparams.pipeline_function, rparams.kwargs)
