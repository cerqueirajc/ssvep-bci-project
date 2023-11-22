from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, replace

import scipy
import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import CCA

from . import runtime_configuration as rc
from .utils import get_harmonic_columns, electrodes_name_to_index, shift_time_dimension

""" TODO

Também preciso adicionar a parte de string com os algoritmos, evitar repetir experimentos já feito (faz sentido?)

Também preciso arrumar a função de Fit do CCA Learner. Decidir se preciso retornar scorado ou não. Fica mais pesado, mas se tiver
mais learners depois dele, o certo é retornar..
"""


@dataclass
class EEGType:
    data: NDArray[np.float64]
    start_time_idx: int
    stop_time_idx: int

    def __post_init__(self):
        if self.data.shape[-2] != self.stop_time_idx - self.start_time_idx:
            raise ValueError("EEG time dimension size does not match start and stop index range")

CorrelationType = NDArray[np.float64]
TransformerReturnType = Union[CorrelationType, EEGType]


class CCALearner(CCA):
    n_components: int

    def correlation(self, X: NDArray, Y: NDArray, n_components: Optional[int] = None) -> list[float]:
        if n_components is None or n_components > self.n_components:
            n_components = self.n_components

        x_projection, y_projection = self.transform(X=X, Y=Y)

        return [
            scipy.stats.pearsonr(x_projection[:,n], y_projection[:,n]).statistic
            for n in range(0, n_components)
        ]

    def fit_correlation(self, X: NDArray, Y: NDArray, n_components: Optional[int] = None):
        return self.fit(X, Y).correlation(X, Y, n_components)


class Transformer(ABC):
    @abstractmethod
    def __call__(self, eeg: TransformerReturnType) -> TransformerReturnType:
        pass

    @abstractmethod
    def fit(self, eeg: TransformerReturnType):
        pass


class NonTrainableTransformer(Transformer):
    def fit(self, eeg: TransformerReturnType):
        return self.__call__(eeg)


class ElectrodesFilter(NonTrainableTransformer):
    def __init__(self, electrodes_index: List[int]) -> None:
        self.electrodes_index = electrodes_index

    def __call__(self, eeg: EEGType) -> EEGType:
        if len(self.electrodes_index) > 0:
            return replace(eeg, data=eeg.data[..., self.electrodes_index])
        return eeg


class TimeFilter(NonTrainableTransformer):
    def __init__(self, start_time_idx: Optional[int], stop_time_idx: Optional[int]) -> None:
        self.start_time_idx = start_time_idx
        self.stop_time_idx = stop_time_idx

    def __call__(self, eeg: EEGType) -> EEGType:
        start_time_idx: int = self.start_time_idx or eeg.start_time_idx
        stop_time_idx: int = self.stop_time_idx or eeg.stop_time_idx

        offset_start_time_idx = start_time_idx - eeg.start_time_idx
        offset_stop_time_idx = stop_time_idx - eeg.start_time_idx

        return EEGType(eeg.data[..., offset_start_time_idx:offset_stop_time_idx, :], start_time_idx, stop_time_idx)


class DummyProjector(NonTrainableTransformer):
    def __call__(self, eeg: EEGType) -> EEGType:
        return replace(eeg, data=eeg.data[..., np.newaxis, :, :])


class CCABase:

    CCA_MAX_ITER = 1000

    def __init__(
            self,
            num_components: int,
            num_harmonics: int,
            num_projections: int
        ) -> None:

        self.num_components: int = num_components
        self.num_harmonics: int = num_harmonics
        self.num_projections: int = num_projections


class CCAModeCorrelation(CCABase, NonTrainableTransformer):

    def __call__(self, eeg: EEGType) -> CorrelationType:
        correlations = np.empty([rc.num_targets, self.num_projections, self.num_components])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = get_harmonic_columns(freq, eeg.start_time_idx, eeg.stop_time_idx, self.num_harmonics)

            for proj in range(self.num_projections):
                cca_model = CCALearner(n_components=self.num_components, max_iter=self.CCA_MAX_ITER, scale=False)
                correlations[target, proj, :] = cca_model.fit_correlation(eeg.data[proj, :, :], harmonic)

        return correlations


class Squeeze(NonTrainableTransformer):
    """This is just a wrapper around Numpy's squeeze function, just to keep code consistent"""
    def __call__(self, eeg: TransformerReturnType) -> TransformerReturnType:
        if isinstance(eeg, EEGType):
            return replace(eeg, data=np.squeeze(eeg.data))
        else:
            return np.squeeze(eeg)


class FilterbankProjector(NonTrainableTransformer):

    def __init__(
        self,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
    ):

        self.fb_num_subband=fb_num_subband
        self.fb_fundamental_freq = fb_fundamental_freq
        self.fb_upper_bound_freq=fb_upper_bound_freq
        self.fb_weight__a=fb_weight__a
        self.fb_weight__b=fb_weight__b

        self.CHEBY_ORDER = 6
        self.CHEBY_MAX_RIPPLE = 0.5
        self.CHEBY_FS = rc.sample_frequency
        self.CHEBY_BAND_TYPE = "bandpass"
        self.FB_BAND_FREQ_TOL = 2
        self.FB_PADDING_TYPE = "odd"

        self.fb_filters = []
        for i in range(self.fb_num_subband):

            freq_low = fb_fundamental_freq * (i + 1)
            freq_high = self.fb_upper_bound_freq
            if freq_low >= freq_high:
                raise ValueError("Number of FB subband is too high, try decreasing it.")

            b, a = scipy.signal.cheby1(
                Wn=[freq_low - self.FB_BAND_FREQ_TOL, freq_high + self.FB_BAND_FREQ_TOL],
                N=self.CHEBY_ORDER,
                rp=self.CHEBY_MAX_RIPPLE,
                btype=self.CHEBY_BAND_TYPE,
                fs=self.CHEBY_FS
            )

            self.fb_filters.append({"b": b, "a": a})


    def __call__(self, eeg: EEGType):
        eeg_fb = (
            eeg.data[..., np.newaxis, :, :]
            .repeat(self.fb_num_subband, axis=-3)
        )

        for filter_idx, filter_params in enumerate(self.fb_filters):
            eeg_fb[..., filter_idx, :, :] = scipy.signal.filtfilt(
                x=eeg_fb[..., filter_idx, :, :],
                b=filter_params["b"],
                a=filter_params["a"],
                padtype=self.FB_PADDING_TYPE,
                axis=-2
            )

        return replace(eeg, data=eeg_fb)


class FilterBankPredictProba(NonTrainableTransformer):

    def __init__(self, fb_num_subband=3, fb_weight__a: float = 1.25, fb_weight__b: float = 0.25) -> None:
        self.fb_weight__a = fb_weight__a
        self.fb_weight__b = fb_weight__b
        self.fb_num_subband = fb_num_subband

        def fb_weight(n):
            a = self.fb_weight__a
            b = self.fb_weight__b
            return n**(-a) + b

        self.fb_weight_array = np.array([fb_weight(n) for n in range(1, self.fb_num_subband + 1)])

    def __call__(self, eeg: CorrelationType) -> CorrelationType:
        return np.power(eeg, 2) @ self.fb_weight_array.T


class SpatioTemporalBank(NonTrainableTransformer):

    def __init__(
        self,
        window_gap=0,
        window_length=10
    ):
        self.window_gap = window_gap
        self.window_length = window_length

        assert self.window_gap >= 0
        assert self.window_length >= 0


    def __call__(self, eeg: EEGType) -> EEGType:

        eeg_with_lags =  (
            eeg.data[..., np.newaxis]
            .repeat(self.window_length + 1, axis=-1)
        )

        for length in range(1, self.window_length + 1): # skip zero, always keep t=0 as original value
            lag = length + self.window_gap
            eeg_with_lags[..., length] = shift_time_dimension(eeg_with_lags[..., length], lag)

        eeg_reshaped = eeg_with_lags.reshape(*eeg_with_lags.shape[:-2], -1) # coalesce last 2 dimensions

        return replace(eeg, data=eeg_reshaped)


class CCAModeFilter(CCABase, Transformer):

    def __call__(self, eeg: EEGType) -> CorrelationType:

        correlations = np.empty([rc.num_targets, self.num_projections, self.num_components])
        for target, freq in enumerate(rc.target_frequencies):
            harmonic = get_harmonic_columns(freq, eeg.start_time_idx, eeg.stop_time_idx, self.num_harmonics)
            for proj in range(self.num_projections):
                correlations[target, proj, :] = self.cca_models[target][proj].correlation(eeg.data[proj, ...], harmonic)

        return correlations


    def fit(self, eeg_tensor: EEGType):
        """Expects eeg_tensor with num_dim=5 and dims=(num_blocks, num_targets, num_projections, num_samples, num_electrodes)"""

        num_blocks = eeg_tensor.data.shape[0]
        num_electrodes = eeg_tensor.data.shape[-1]

        self.cca_models = defaultdict(lambda : {})

        for target, freq in enumerate(rc.target_frequencies):
            eeg_blocks = (
                eeg_tensor.data[:, target, ...]                     # num_blocks, num_projections, num_samples, num_electrodes
                .transpose([1, 0, 2, 3])                            # num_projections, num_blocks, num_samples, num_electrodes
                .reshape(self.num_projections, -1, num_electrodes)  # num_projections, num_blocks*num_samples, num_electrodes
            )

            harmonic = get_harmonic_columns(freq, eeg_tensor.start_time_idx, eeg_tensor.stop_time_idx, self.num_harmonics)

            harmonic_concatenated = (
                # create new dummy dimension that will be expanded
                harmonic[np.newaxis, :, :]
                # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .repeat(num_blocks, axis=0)
                # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
                .reshape(-1, self.num_harmonics * 2)
            )

            for proj in range(self.num_projections):
                cca_model = CCALearner(n_components=self.num_components, max_iter=self.CCA_MAX_ITER, scale=False)
                cca_model.fit(eeg_blocks[proj, ...], harmonic_concatenated)
                self.cca_models[target][proj] = cca_model

        return self.__call__(replace(eeg_tensor, data=eeg_tensor.data[0,0,...]))


def chain_call_transformers(arg: TransformerReturnType, *funcs: Callable) -> TransformerReturnType:
    result = arg
    for f in funcs:
        result = f(result)
    return result


class SSVEPAlgorithm:
    pipeline: Tuple[Transformer, ...]

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, eeg: EEGType) -> Tuple[np.intp, CorrelationType]:
        res: Any = chain_call_transformers(eeg, *self.pipeline)
        return np.argmax(res), res

    def fit(self, eeg_tensor: EEGType) -> TransformerReturnType:
        res = chain_call_transformers(eeg_tensor, *[p.fit for p in self.pipeline])
        return res


class StandardCCA(SSVEPAlgorithm):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        # num_components=1,
        num_harmonics=3,
    ) -> None:

        self.start_time_idx = start_time_idx
        self.stop_time_idx = stop_time_idx
        self.electrodes_name = electrodes_name
        self.electrodes_index = electrodes_name_to_index(electrodes_name)
        self.num_components = 1 # TODO implement multiple components
        self.num_harmonics = num_harmonics
        self.initialize_pipeline()

    def initialize_pipeline(self) -> None:
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(self.start_time_idx, self.stop_time_idx),
            DummyProjector(),
            CCAModeCorrelation(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics,
                num_projections=1
            ),
            Squeeze()
        )


class FilterbankCCA(SSVEPAlgorithm):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        num_harmonics=3,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
    ):
        self.start_time_idx = start_time_idx
        self.stop_time_idx = stop_time_idx
        self.electrodes_name = electrodes_name
        self.electrodes_index = electrodes_name_to_index(electrodes_name)
        self.num_components = 1 # TODO implement multiple components
        self.num_harmonics = num_harmonics
        self.fb_num_subband=fb_num_subband
        self.fb_fundamental_freq = fb_fundamental_freq
        self.fb_upper_bound_freq=fb_upper_bound_freq
        self.fb_weight__a=fb_weight__a
        self.fb_weight__b=fb_weight__b

        self.CHEBY_ORDER = 6
        self.CHEBY_MAX_RIPPLE = 0.5
        self.CHEBY_FS = rc.sample_frequency
        self.CHEBY_BAND_TYPE = "bandpass"
        self.FB_BAND_FREQ_TOL = 2
        self.FB_PADDING_TYPE = "odd"

        self.initialize_pipeline()


    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(self.start_time_idx, self.stop_time_idx),
            FilterbankProjector(
                fb_num_subband=self.fb_num_subband,
                fb_fundamental_freq=self.fb_fundamental_freq,
                fb_upper_bound_freq=self.fb_upper_bound_freq,
                fb_weight__a=self.fb_weight__a,
                fb_weight__b=self.fb_weight__b
            ),
            CCAModeCorrelation(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics,
                num_projections=self.fb_num_subband
            ),
            Squeeze(),
            FilterBankPredictProba(
                fb_num_subband=self.fb_num_subband,
                fb_weight__a=self.fb_weight__a,
                fb_weight__b=self.fb_weight__b
            )
        )


class SpatioTemporalCCA(SSVEPAlgorithm):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        num_harmonics=3,
        window_gap=0,
        window_length=5
    ):
        if window_length + window_gap > start_time_idx:
            raise ValueError("Value of start_time_idx must be smaller or equal to window_length + window_gap")

        self.start_time_idx = start_time_idx
        self.stop_time_idx = stop_time_idx
        self.electrodes_name = electrodes_name
        self.electrodes_index = electrodes_name_to_index(electrodes_name)
        self.num_components = 1 # TODO implement multiple components
        self.num_harmonics = num_harmonics
        self.window_gap=window_gap
        self.window_length=window_length
        self.initialize_pipeline()


    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(None, self.stop_time_idx),
            SpatioTemporalBank(window_gap=self.window_gap, window_length = self.window_length),
            TimeFilter(max([self.window_gap + self.window_length, self.start_time_idx]), None),
            DummyProjector(),
            CCAModeCorrelation(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics,
                num_projections=1
            ),
            Squeeze()
        )


class StandardCCAFilter(SSVEPAlgorithm):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        num_harmonics=3,
    ) -> None:

        self.start_time_idx = start_time_idx
        self.stop_time_idx = stop_time_idx
        self.electrodes_name = electrodes_name
        self.electrodes_index = electrodes_name_to_index(electrodes_name)
        self.num_components = 1 # TODO implement multiple components
        self.num_harmonics = num_harmonics
        self.initialize_pipeline()

    def initialize_pipeline(self) -> None:
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(self.start_time_idx, self.stop_time_idx),
            DummyProjector(),
            CCAModeFilter(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics,
                num_projections=1
            ),
            Squeeze()
        )


class FilterbankCCAFilter(FilterbankCCA):
    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(self.start_time_idx, self.stop_time_idx),
            FilterbankProjector(
                fb_num_subband=self.fb_num_subband,
                fb_fundamental_freq=self.fb_fundamental_freq,
                fb_upper_bound_freq=self.fb_upper_bound_freq,
                fb_weight__a=self.fb_weight__a,
                fb_weight__b=self.fb_weight__b
            ),
            CCAModeFilter(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics,
                num_projections=self.fb_num_subband
            ),
            Squeeze(),
            FilterBankPredictProba(
                fb_num_subband=self.fb_num_subband,
                fb_weight__a=self.fb_weight__a,
                fb_weight__b=self.fb_weight__b
            )
        )
