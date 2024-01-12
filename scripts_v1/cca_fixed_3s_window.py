import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

import sys

    
def main():
    
    learner = learners.CCAFixedCoefficients(
        electrodes_name=parameters.electrode_list_fbcca,
        start_time_index=125,
        stop_time_index=875,
    )

    fit_pipeline = pipelines.k_fold_predict

    pipelines.eval_all_subjects_and_save_pipeline(
        learner_obj=learner,
        fit_pipeline=fit_pipeline,
        dataset_root_path="../dataset_chines",
        output_folder="results/" + sys.argv[0].split(".py")[0]
    )


if __name__ == "__main__":
    main()
