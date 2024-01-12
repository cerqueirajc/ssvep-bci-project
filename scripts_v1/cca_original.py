import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

import sys

    
def main():
    
    learner = learners.CCASingleComponent(
        electrodes_name=parameters.electrode_list_fbcca,
        start_time_index=0,
        stop_time_index=1500,
    )

    fit_pipeline = pipelines.test_fit_predict

    pipelines.eval_all_subjects_and_save_pipeline(
        learner_obj=learner,
        fit_pipeline=fit_pipeline,
        dataset_root_path="../dataset_chines",
        output_folder="results/" + sys.argv[0].split(".py")[0]
    )


if __name__ == "__main__":
    main()
