import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

from scripts_new.configurations import (
    RunParams, run_exector
)

START_TIME_INDEX = 125
STOP_TIME_INDEX = 625
TIME_SUFFIX = "[2s]"

# START_TIME_INDEX = 125
# STOP_TIME_INDEX = 875
# TIME_SUFFIX = "[3s]"
    
RUN_PARAMS = [
    RunParams(
        f"01_Default_CCA_{TIME_SUFFIX}",
        pipelines.test_fit_predict,
        learners.CCASingleComponent(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
        ),
    ),
    RunParams(
        f"02_SS-CCA_(3,1)_{TIME_SUFFIX}",
        pipelines.test_fit_predict,
        learners.CCASpatioTemporal(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=3,
            window_length=1,
        ),
    ),
    RunParams(
        f"03_SS-CCA_(0,5)_{TIME_SUFFIX}",
        pipelines.test_fit_predict,
        learners.CCASpatioTemporal(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=0,
            window_length=5,
        ),
    ),
    RunParams(
        f"04_Filterbank_CCA_{TIME_SUFFIX}",
        pipelines.test_fit_predict,
        learners.FilterbankCCA(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            fb_num_subband=10,
            fb_fundamental_freq=8,
            fb_upper_bound_freq=88,
        ),
    ),
    RunParams(
        f"05_Filterbank_SS-CCA_(3,1)_{TIME_SUFFIX}",
        pipelines.test_fit_predict,
        learners.FBSpatioTemporalCCA(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=3,
            window_length=1,
            fb_num_subband=10,
            fb_fundamental_freq=8,
            fb_upper_bound_freq=88,
        ),
    ),

    RunParams(
        f"07_Default_CCA_Fixed_{TIME_SUFFIX}",
        pipelines.k_fold_predict,
        learners.CCAFixedCoefficients(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
        ),
    ),
    RunParams(
        f"08_SS-CCA_(3,1)_Fixed_{TIME_SUFFIX}",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=3,
            window_length=1,
        ),
    ),
    RunParams(
        f"09_SS-CCA_(0,5)_Fixed_{TIME_SUFFIX}",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=0,
            window_length=5,
        ),
    ),
    RunParams(
        f"10_Filterbank_CCA_Fixed_{TIME_SUFFIX}",
        pipelines.k_fold_predict,
        learners.FBCCAFixedCoefficients(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            fb_num_subband=10,
            fb_fundamental_freq=8,
            fb_upper_bound_freq=88,
        ),
    ),
    RunParams(
        f"11_Filterbank_SS-CCA_(3,1)_Fixed_{TIME_SUFFIX}",
        pipelines.k_fold_predict,
        learners.FBSpatioTemporalCCAFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=3,
            window_length=1,
            fb_num_subband=10,
            fb_fundamental_freq=8,
            fb_upper_bound_freq=88,
        ),
    ),
    RunParams(
        f"12_Filterbank_SS-CCA_(0,5)_Fixed_{TIME_SUFFIX}",
        pipelines.k_fold_predict,
        learners.FBSpatioTemporalCCAFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=0,
            window_length=5,
            fb_num_subband=10,
            fb_fundamental_freq=8,
            fb_upper_bound_freq=88,
        ),
    ),
    RunParams(
        f"06_Filterbank_SS-CCA_(0,5)_{TIME_SUFFIX}",
        pipelines.test_fit_predict,
        learners.FBSpatioTemporalCCA(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=0,
            window_length=5,
            fb_num_subband=10,
            fb_fundamental_freq=8,
            fb_upper_bound_freq=88,
        ),
    ),
]

if __name__ == "__main__":
    for i, run_params in enumerate(RUN_PARAMS):
        print(f"Running... params number {i + 1}/{len(RUN_PARAMS)}: {run_params.name}")    
        run_exector(run_params)
