import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

from scripts_new.configurations import (
    START_TIME_INDEX, STOP_TIME_INDEX,
    RunParams, run_exector
)


FIR_CCA_WINDOW_LENGTH_LIST = list(range(20))

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
    for window_length in FIR_CCA_WINDOW_LENGTH_LIST
]

if __name__ == "__main__":
    for i, run_params in enumerate(RUN_PARAMS):
        print(f"Running... params number {i + 1}/{len(RUN_PARAMS)}: {run_params.name}")    
        run_exector(run_params)
