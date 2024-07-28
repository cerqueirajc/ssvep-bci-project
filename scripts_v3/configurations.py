from typing import Tuple
from toolz import merge

from ssvepcca import parameters
from ssvepcca.pipelines import test_fit_predict, k_fold_predict, k_fold_predict_alt
from ssvepcca.algorithms import (
    FilterbankCCAMulticlass, SpatioTemporalCCAMulticlass, StandardCCA, FilterbankCCA,
    SpatioTemporalCCA, FBSpatioTemporalCCA, StandardCCAFilter,
    FilterbankCCAFilter, SpatioTemporalCCAFilter,
    FBSpatioTemporalCCAFilter, StandardCCAMulticlass
)

from routine import ExperimentParams


# Constants

PARAMS_CCA_SINGLE_COMPONENT = {
    "electrodes_name": parameters.electrode_list_fbcca,
    "num_harmonics": 5,
}

PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT = merge(
    PARAMS_CCA_SINGLE_COMPONENT,
    {
        "window_gap": 2,
        "window_length": 1,
    }
)

PARAMS_FIR_CCA_DEFAULT = merge(
    PARAMS_CCA_SINGLE_COMPONENT,
    {
        "window_gap": 0,
        "window_length": 4,
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

SS_CCA_WINDOW_GAP_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
SS_WINDOW_LENGTH = 1

FIR_CCA_WINDOW_GAP = 0
FIR_CCA_WINDOW_LENGTH_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]

FIR_GRIDSEARCH_GAP    = [0, 1, 2, 3, 4, 5, 6]
FIR_GRIDSEARCH_LENGTH = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Functions

def ss_fir_params_str(window_gap, window_length):
    return f"G{window_gap}__L{window_length}"


# Experiment params

experiments_correlation: Tuple[ExperimentParams, ...] = (
    ExperimentParams(StandardCCA, test_fit_predict, PARAMS_CCA_SINGLE_COMPONENT, "default"),
    ExperimentParams(SpatioTemporalCCA, test_fit_predict, PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT, "default_ss_cca"),
    ExperimentParams(SpatioTemporalCCA, test_fit_predict, PARAMS_FIR_CCA_DEFAULT, "default_fir_cca"),
    ExperimentParams(FilterbankCCA, test_fit_predict, PARAMS_FILTERBANK_CCA, "default"),
    # ExperimentParams(FBSpatioTemporalCCA, test_fit_predict, PARAMS_FB_SS_CCA_DEFAULT, "default_fb_ss_cca"),
    # ExperimentParams(FBSpatioTemporalCCA, test_fit_predict, PARAMS_FB_FIR_CCA_DEFAULT, "default_fb_fir_cca"),
)


experiments_filter: Tuple[ExperimentParams, ...] = (
    ExperimentParams(StandardCCAFilter, k_fold_predict, PARAMS_CCA_SINGLE_COMPONENT, "default"),
    ExperimentParams(SpatioTemporalCCAFilter, k_fold_predict, PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT, "default_ss_cca"),
    ExperimentParams(SpatioTemporalCCAFilter, k_fold_predict, PARAMS_FIR_CCA_DEFAULT, "default_fir_cca"),
    ExperimentParams(FilterbankCCAFilter, k_fold_predict, PARAMS_FILTERBANK_CCA, "default"),
    # ExperimentParams(FBSpatioTemporalCCAFilter, k_fold_predict, PARAMS_FB_SS_CCA_DEFAULT, "default_fb_ss_cca"),
    # ExperimentParams(FBSpatioTemporalCCAFilter, k_fold_predict, PARAMS_FB_FIR_CCA_DEFAULT, "default_fb_fir_cca"),
)

kfold_3_blocks = k_fold_predict_alt(train_num_blocks=3)
experiments_filter_short_training_3: Tuple[ExperimentParams, ...] = (
    ExperimentParams(StandardCCAFilter, kfold_3_blocks, PARAMS_CCA_SINGLE_COMPONENT, "default__3_blocks"),
    ExperimentParams(SpatioTemporalCCAFilter, kfold_3_blocks, PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT, "default_ss_cca__3_blocks"),
    ExperimentParams(SpatioTemporalCCAFilter, kfold_3_blocks, PARAMS_FIR_CCA_DEFAULT, "default_fir_cca__3_blocks"),
    ExperimentParams(FilterbankCCAFilter, kfold_3_blocks, PARAMS_FILTERBANK_CCA, "default__3_blocks"),
    # ExperimentParams(FBSpatioTemporalCCAFilter, kfold_3_blocks, PARAMS_FB_SS_CCA_DEFAULT, "default_fb_ss_cca__3_blocks"),
    # ExperimentParams(FBSpatioTemporalCCAFilter, kfold_3_blocks, PARAMS_FB_FIR_CCA_DEFAULT, "default_fb_fir_cca__3_blocks"),
)

kfold_1_blocks = k_fold_predict_alt(train_num_blocks=1)
experiments_filter_short_training_1: Tuple[ExperimentParams, ...] = (
    ExperimentParams(StandardCCAFilter, kfold_1_blocks, PARAMS_CCA_SINGLE_COMPONENT, "default__1_blocks"),
    ExperimentParams(SpatioTemporalCCAFilter, kfold_1_blocks, PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT, "default_ss_cca__1_blocks"),
    ExperimentParams(SpatioTemporalCCAFilter, kfold_1_blocks, PARAMS_FIR_CCA_DEFAULT, "default_fir_cca__1_blocks"),
    ExperimentParams(FilterbankCCAFilter, kfold_1_blocks, PARAMS_FILTERBANK_CCA, "default__1_blocks"),
    # ExperimentParams(FBSpatioTemporalCCAFilter, kfold_3_blocks, PARAMS_FB_SS_CCA_DEFAULT, "default_fb_ss_cca__3_blocks"),
    # ExperimentParams(FBSpatioTemporalCCAFilter, kfold_3_blocks, PARAMS_FB_FIR_CCA_DEFAULT, "default_fb_fir_cca__3_blocks"),
)

experiments_new_cca_multiclass: Tuple[ExperimentParams, ...] = (
    ExperimentParams(StandardCCAMulticlass, k_fold_predict, PARAMS_CCA_SINGLE_COMPONENT, "default"),
    ExperimentParams(SpatioTemporalCCAMulticlass, k_fold_predict, PARAMS_SPATIO_TEMPORAL_CCA_DEFAULT, "default_ss_cca"),
    ExperimentParams(SpatioTemporalCCAMulticlass, k_fold_predict, PARAMS_FIR_CCA_DEFAULT, "default_fir_cca"),
    ExperimentParams(FilterbankCCAMulticlass, k_fold_predict, PARAMS_FILTERBANK_CCA, "default"),
)

### Grid search parameters

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
