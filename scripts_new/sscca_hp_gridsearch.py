import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

import sys
from dataclasses import dataclass


START_TIME_INDEX = 125
STOP_TIME_INDEX = 875
OUTPUT_ROOT_FOLDER = "results_ssccafixed_hp/"


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

SS_CCA_WINDOW_LENGTH_LIST = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40]

RUN_PARAMS = [
    RunParams(
        f"SS-CCA (0, {window_length}) Fixed [2s]",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=0,
            window_length=window_length,
        ),
    )
    for window_length in SS_CCA_WINDOW_LENGTH_LIST
]

if __name__ == "__main__":
    for i, run_params in enumerate(RUN_PARAMS):
        print(f"Running... params number {i + 1}/{len(RUN_PARAMS)}: {run_params.name}")    
        run_exector(run_params)
