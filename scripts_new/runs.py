import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

import sys
from dataclasses import dataclass


START_TIME_INDEX = 125
STOP_TIME_INDEX = 875
OUTPUT_ROOT_FOLDER = "results_new/"


@dataclass
class RunParams:
    """Class for keeping an experiment parameters"""
    name: str
    pipeline_function: callable
    learner_obj: object


def run_exector(run_params: RunParams):
    pipelines.eval_all_subjects_and_save_pipeline(
        learner_obj=run_params.learner_obj,
        fit_pipeline=run_params.pipeline_function,
        dataset_root_path="../dataset_chines",
        output_folder=OUTPUT_ROOT_FOLDER + sys.argv[0].split(".py")[0] + "_" + run_params.name
    )

    
RUN_PARAMS = [
    RunParams(
        "Default CCA [2s]",
        pipelines.test_fit_predict,
        learners.CCASingleComponent(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
        ),
    ),
    RunParams(
        "SS-CCA (5, 1) [2s]",
        pipelines.test_fit_predict,
        learners.CCASpatioTemporal(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=5,
            window_length=1,
        ),
    ),
    RunParams(
        "SS-CCA (0, 9) [2s]",
        pipelines.test_fit_predict,
        learners.CCASpatioTemporal(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=0,
            window_length=9,
        ),
    ),
    RunParams(
        "Filterbank CCA [2s]",
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
        "Filterbank SS-CCA (2, 1) [2s]",
        pipelines.test_fit_predict,
        learners.FBSpatioTemporalCCA(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=2,
            window_length=1,
            fb_num_subband=10,
            fb_fundamental_freq=8,
            fb_upper_bound_freq=88,
        ),
    ),
    RunParams(
        "Default CCA Fixed [2s]",
        pipelines.k_fold_predict,
        learners.CCAFixedCoefficients(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
        ),
    ),
    RunParams(
        "SS-CCA (5, 1) Fixed [2s]",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=5,
            window_length=1,
        ),
    ),
    RunParams(
        "SS-CCA (0, 9) Fixed [2s]",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=0,
            window_length=9,
        ),
    ),
    RunParams(
        "Filterbank CCA Fixed [2s]",
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
        "Filterbank SS-CCA (2, 1) Fixed [2s]",
        pipelines.k_fold_predict,
        learners.FBSpatioTemporalCCAFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=5,
            window_gap=2,
            window_length=1,
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
