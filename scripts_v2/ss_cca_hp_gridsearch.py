import ssvepcca.pipelines as pipelines
import ssvepcca.learners as learners
import ssvepcca.parameters as parameters

from scripts_new.configurations import (
    START_TIME_INDEX, STOP_TIME_INDEX,
    RunParams, run_exector
)


SS_CCA_WINDOW_LENGTH = 1
SS_CCA_WINDOW_GAP_LIST = list(range(31))

RUN_PARAMS = [
    RunParams(
        f"SS-CCA_({window_gap},{SS_CCA_WINDOW_LENGTH})__[2s]",
        pipelines.test_fit_predict,
        learners.CCASpatioTemporal(
            electrodes_name=parameters.electrode_list_fbcca,
            start_time_index=START_TIME_INDEX,
            stop_time_index=STOP_TIME_INDEX,
            num_harmonics=3,
            window_gap=window_gap,
            window_length=SS_CCA_WINDOW_LENGTH,
        ),
    )
    for window_gap in SS_CCA_WINDOW_GAP_LIST
]

if __name__ == "__main__":
    for i, run_params in enumerate(RUN_PARAMS):
        print(f"Running... params number {i + 1}/{len(RUN_PARAMS)}: {run_params.name}")    
        run_exector(run_params)
