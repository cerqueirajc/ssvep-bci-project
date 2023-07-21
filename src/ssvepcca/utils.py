import numpy as np
import pandas as pd
import scipy

from functools import cache

from ssvepcca.definitions import SAMPLE_T, NUM_BLOCKS, NUM_TARGETS, NUM_SAMPLES, NUM_ELECTRODES, ELECTRODE_INDEX


def check_input_data(data):
    assert data.shape == (NUM_BLOCKS, NUM_TARGETS, NUM_SAMPLES, NUM_ELECTRODES)


def check_result_data(data):
    assert data.shape == (NUM_BLOCKS, NUM_TARGETS), f"Input result data dimensions does not match: {data.shape}"


def _get_time_column(start_time_index, stop_time_index):
    return np.arange(start_time_index, stop_time_index, 1).reshape(1, -1) * SAMPLE_T


@cache
def get_harmonic_columns(
    frequency,
    start_time_index=0,
    stop_time_index=1500,
    num_harmonics=3
):
    time_col = _get_time_column(start_time_index, stop_time_index)
    harmonics = []

    for h in range(1, num_harmonics + 1):
        harmonics.append(np.sin(time_col * 2 * scipy.pi * (frequency * h)))
        harmonics.append(np.cos(time_col * 2 * scipy.pi * (frequency * h)))

    return np.concatenate(harmonics).T


def electrodes_name_to_index(electrodes):
     return [ELECTRODE_INDEX[electrode_name] for electrode_name in electrodes]


def eval_accuracy(result):
    check_result_data(result)

    count = 0
    for b in range(NUM_BLOCKS):
        for t in range(NUM_TARGETS):
            if result[b, t] == t:
                count += 1

    return [count, count/(NUM_BLOCKS * NUM_TARGETS)]


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

