import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

from scripts_new.configurations import (
    START_TIME_INDEX, STOP_TIME_INDEX,
    RunParams, run_exector
)


SS_CCA_WINDOW_GAP_LIST = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40]

RUN_PARAMS = [
    RunParams(
        f"SS-CCA (0, {window_gap}) Fixed [2s]",
        pipelines.k_fold_predict,
        learners.CCASpatioTemporalFixed(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=window_gap,
            window_length=1,
        ),
    )
    for window_gap in SS_CCA_WINDOW_GAP_LIST
]

if __name__ == "__main__":
    for i, run_params in enumerate(RUN_PARAMS):
        print(f"Running... params number {i + 1}/{len(RUN_PARAMS)}: {run_params.name}")    
        run_exector(run_params)
