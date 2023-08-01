OUTPUT_ROOT_FOLDER = "results/"
START_TIME_INDEX = 125
STOP_TIME_INDEX = 875

from dataclasses import dataclass
import ssvepcca.pipelines as pipelines
import sys


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
        output_folder=OUTPUT_ROOT_FOLDER + sys.argv[0].split(".py")[0] + "/" + run_params.name
    )
