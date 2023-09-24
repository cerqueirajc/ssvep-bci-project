import numpy as np
import pandas as pd
import scipy
from functools import cache

from . import runtime_configuration as rc


def check_input_data(data: np.ndarray) -> None:
    assert data.shape == (rc.num_blocks, rc.num_targets, rc.num_samples, rc.num_electrodes)


def check_result_data(data: np.ndarray) -> None:
    assert data.shape == (rc.num_blocks, rc.num_targets), f"Input result data dimensions does not match: {data.shape}"


def get_time_column(start_time_index: int, stop_time_index: int) -> np.ndarray:
    return np.arange(start_time_index, stop_time_index, 1).reshape(1, -1) * (1 / rc.sample_frequency)


@cache
def get_harmonic_columns(
    frequency: float,
    start_time_index: int = 0,
    stop_time_index: int = 1500,
    num_harmonics: int = 3
) -> np.ndarray:
    time_col = get_time_column(start_time_index, stop_time_index)
    harmonics = []

    for h in range(1, num_harmonics + 1):
        harmonics.append(np.sin(time_col * 2 * np.pi * (frequency * h)))
        harmonics.append(np.cos(time_col * 2 * np.pi * (frequency * h)))

    return np.concatenate(harmonics).T


def electrodes_name_to_index(electrodes):
     return [rc.electrodes[electrode_name] for electrode_name in electrodes]


def eval_accuracy(result):
    check_result_data(result)

    count = 0
    for b in range(rc.num_blocks):
        for t in range(rc.num_targets):
            if result[b, t] == t:
                count += 1

    return [count, count/(rc.num_blocks * rc.num_targets)]


def load_mat_data_array(mat_path):
    mat = scipy.io.loadmat(mat_path)
    return mat["data"].astype(float).T


def load_mat_to_pandas(mat_path):

    mat = scipy.io.loadmat(mat_path)["data"]

    temp_list = []

    for block in range(6):
        for target in range(40):
            temp_list.append(
                pd.DataFrame(mat[:, :, target, block].T)
                    .add_prefix("electrode_")
                    .assign(target=target)
                    .assign(block=block)
                    .reset_index()
                    .rename(columns={"index": "time_index"})
                    .assign(time_ms=lambda df: (df["time_index"] + 1) * 4)
            )

    return pd.concat(temp_list)


def shift_first_dim(arr, num):
    """
    Shifts the values of a tensor of time series, assuming the first dimension is the time index.
    """

    arr=np.roll(arr, num, axis=0)
    if num < 0:
        arr[-num:, ...] = np.nan
    elif num > 0:
        arr[:num, ...] = np.nan
    return arr

