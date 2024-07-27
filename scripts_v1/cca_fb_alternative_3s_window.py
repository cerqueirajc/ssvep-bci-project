import ssvepcca.pipelines as pipelines
import ssvepcca.deprecated_learners as deprecated_learners
import ssvepcca.parameters as parameters

import sys

    
def main():
    
    learner = deprecated_learners.AlternativeFBCCA(
        electrodes_name=parameters.electrode_list_fbcca,
        start_time_index=125,
        stop_time_index=875,
        num_harmonics=3,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
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
