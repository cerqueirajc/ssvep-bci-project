import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

from scripts_new.configurations import (
    START_TIME_INDEX, STOP_TIME_INDEX,
    RunParams, run_exector
)

    
RUN_PARAMS = [
    # RunParams(
    #     "01_Default_CCA_[2s]",
    #     pipelines.test_fit_predict,
    #     learners.CCASingleComponent(
    #         electrodes_name=parameters.electrode_list_fbcca,
    #         start_time_index=START_TIME_INDEX,
    #         stop_time_index=STOP_TIME_INDEX,
    #         num_harmonics=3,
    #     ),
    # ),
    # RunParams(
    #     "02_SS-CCA_(3,1)_[2s]",
    #     pipelines.test_fit_predict,
    #     learners.CCASpatioTemporal(
    #         electrodes_name=parameters.electrode_list_fbcca,
    #         start_time_index=START_TIME_INDEX,
    #         stop_time_index=STOP_TIME_INDEX,
    #         num_harmonics=3,
    #         window_gap=3,
    #         window_length=1,
    #     ),
    # ),
    # RunParams(
    #     "03_SS-CCA_(0,5)_[2s]",
    #     pipelines.test_fit_predict,
    #     learners.CCASpatioTemporal(
    #         electrodes_name=parameters.electrode_list_fbcca,
    #         start_time_index=START_TIME_INDEX,
    #         stop_time_index=STOP_TIME_INDEX,
    #         num_harmonics=3,
    #         window_gap=0,
    #         window_length=5,
    #     ),
    # ),
    # RunParams(
    #     "04_Filterbank_CCA_[2s]",
    #     pipelines.test_fit_predict,
    #     learners.FilterbankCCA(
    #         electrodes_name=parameters.electrode_list_fbcca,
    #         start_time_index=START_TIME_INDEX,
    #         stop_time_index=STOP_TIME_INDEX,
    #         num_harmonics=5,
    #         fb_num_subband=10,
    #         fb_fundamental_freq=8,
    #         fb_upper_bound_freq=88,
    #     ),
    # ),
    # RunParams(
    #     "05_Filterbank_SS-CCA_(3,1)_[2s]",
    #     pipelines.test_fit_predict,
    #     learners.FBSpatioTemporalCCA(
    #         electrodes_name=parameters.electrode_list_fbcca,
    #         start_time_index=START_TIME_INDEX,
    #         stop_time_index=STOP_TIME_INDEX,
    #         num_harmonics=5,
    #         window_gap=3,
    #         window_length=1,
    #         fb_num_subband=10,
    #         fb_fundamental_freq=8,
    #         fb_upper_bound_freq=88,
    #     ),
    # ),

    RunParams(
        "07_Default_CCA_Fixed_[2s]",
        pipelines.k_fold_predict,
        learners.CCAFixedCoefficients(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
        ),
    ),
    RunParams(
        "08_SS-CCA_(3,1)_Fixed_[2s]",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=3,
            window_length=1,
        ),
    ),
    RunParams(
        "09_SS-CCA_(0,5)_Fixed_[2s]",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=0,
            window_length=5,
        ),
    ),
    RunParams(
        "10_Filterbank_CCA_Fixed_[2s]",
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
        "11_Filterbank_SS-CCA_(3,1)_Fixed_[2s]",
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
        "12_Filterbank_SS-CCA_(0,5)_Fixed_[2s]",
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
        "06_Filterbank_SS-CCA_(0,5)_[2s]",
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
