from typing import Any, Callable, Tuple
import numpy as np

from .utils import electrodes_name_to_index
from .transformers import (
    TransformerReturnType,
    EEGType,
    CorrelationType,
    Transformer,
    ElectrodesFilter,
    TimeFilter,
    SpatioTemporalBank,
    CCAModeCorrelation,
    CCAModeFilter,
    SpatioTemporalBank,
    DummyProjector,
    FilterbankProjector,
    Squeeze,
    FilterBankPredictProba
)


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
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
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
                num_harmonics=self.num_harmonics
            ),
            Squeeze()
        )


class FilterbankCCA(SSVEPAlgorithm):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
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
        self.initialize_pipeline()


    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(self.start_time_idx, self.stop_time_idx),
            FilterbankProjector(
                fb_num_subband=self.fb_num_subband,
                fb_fundamental_freq=self.fb_fundamental_freq,
                fb_upper_bound_freq=self.fb_upper_bound_freq,
            ),
            CCAModeCorrelation(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics,
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
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
        window_gap=0,
        window_length=5
    ):
        if window_length + window_gap > start_time_idx:
            raise ValueError("Value of start_time_idx must be bigger or equal to window_length + window_gap")

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
            SpatioTemporalBank(window_gap=self.window_gap, window_length=self.window_length),
            TimeFilter(self.start_time_idx, None),
            DummyProjector(),
            CCAModeCorrelation(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics
            ),
            Squeeze()
        )


class FBSpatioTemporalCCA(SSVEPAlgorithm):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
        window_gap=0,
        window_length=5
    ):
        if window_length + window_gap > start_time_idx:
            print(window_length, window_gap, start_time_idx)
            raise ValueError("Value of start_time_idx must be bigger or equal to window_length + window_gap")

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

        self.window_gap=window_gap
        self.window_length=window_length
        self.initialize_pipeline()


    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(None, self.stop_time_idx),
            SpatioTemporalBank(window_gap=self.window_gap, window_length = self.window_length),
            TimeFilter(self.start_time_idx, None),
            FilterbankProjector(
                fb_num_subband=self.fb_num_subband,
                fb_fundamental_freq=self.fb_fundamental_freq,
                fb_upper_bound_freq=self.fb_upper_bound_freq,
            ),
            CCAModeCorrelation(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics
            ),
            Squeeze(),
            FilterBankPredictProba(
                fb_num_subband=self.fb_num_subband,
                fb_weight__a=self.fb_weight__a,
                fb_weight__b=self.fb_weight__b
            )
        )


class StandardCCAFilter(StandardCCA):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
    ) -> None:
        
        self.fit_start_time_idx = fit_start_time_idx
        self.fit_stop_time_idx = fit_stop_time_idx
        super().__init__(
            electrodes_name,
            start_time_idx,
            stop_time_idx,
            num_harmonics
        )

    def initialize_pipeline(self) -> None:
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(
                start_time_idx=self.start_time_idx,
                stop_time_idx=self.stop_time_idx,
                fit_start_time_idx=self.fit_start_time_idx,
                fit_stop_time_idx=self.fit_stop_time_idx,
            ),
            DummyProjector(),
            CCAModeFilter(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics
            ),
            Squeeze()
        )


class FilterbankCCAFilter(FilterbankCCA):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
    ) -> None:

        self.fit_start_time_idx = fit_start_time_idx
        self.fit_stop_time_idx = fit_stop_time_idx
        super().__init__(
            electrodes_name=electrodes_name,
            start_time_idx=start_time_idx,
            stop_time_idx=stop_time_idx,
            num_harmonics=num_harmonics,
            fb_num_subband=fb_num_subband,
            fb_fundamental_freq=fb_fundamental_freq,
            fb_upper_bound_freq=fb_upper_bound_freq,
            fb_weight__a=fb_weight__a,
            fb_weight__b=fb_weight__b,
        )

    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(
                start_time_idx=self.start_time_idx,
                stop_time_idx=self.stop_time_idx,
                fit_start_time_idx=self.fit_start_time_idx,
                fit_stop_time_idx=self.fit_stop_time_idx,
            ),
            FilterbankProjector(
                fb_num_subband=self.fb_num_subband,
                fb_fundamental_freq=self.fb_fundamental_freq,
                fb_upper_bound_freq=self.fb_upper_bound_freq,
            ),
            CCAModeFilter(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics
            ),
            Squeeze(),
            FilterBankPredictProba(
                fb_num_subband=self.fb_num_subband,
                fb_weight__a=self.fb_weight__a,
                fb_weight__b=self.fb_weight__b
            )
        )


class SpatioTemporalCCAFilter(SpatioTemporalCCA):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
        window_gap=0,
        window_length=5
    ) -> None:
        
        if fit_start_time_idx is not None and window_length + window_gap > fit_start_time_idx:
            raise ValueError("Value of fit_start_time_idx must be bigger or equal to window_length + window_gap")

        self.fit_start_time_idx = fit_start_time_idx
        self.fit_stop_time_idx = fit_stop_time_idx
        super().__init__(
            electrodes_name=electrodes_name,
            start_time_idx=start_time_idx,
            stop_time_idx=stop_time_idx,
            num_harmonics=num_harmonics,
            window_gap=window_gap,
            window_length=window_length
        )


    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(
                start_time_idx=None,
                stop_time_idx=self.stop_time_idx,
                fit_start_time_idx=None,
                fit_stop_time_idx=self.fit_stop_time_idx
            ),
            SpatioTemporalBank(window_gap=self.window_gap, window_length = self.window_length),
            TimeFilter(
                start_time_idx=self.start_time_idx,
                stop_time_idx=None,
                fit_start_time_idx=self.fit_start_time_idx,
                fit_stop_time_idx=None,
            ),
            DummyProjector(),
            CCAModeFilter(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics,
            ),
            Squeeze()
        )


class FBSpatioTemporalCCAFilter(SSVEPAlgorithm):

    def __init__(
        self,
        electrodes_name=[],
        start_time_idx=0,
        stop_time_idx=1500,
        fit_start_time_idx=None,
        fit_stop_time_idx=None,
        num_harmonics=5,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
        window_gap=0,
        window_length=5
    ):
        if window_length + window_gap > start_time_idx:
            raise ValueError("Value of start_time_idx must be bigger or equal to window_length + window_gap")

        if fit_start_time_idx is not None and window_length + window_gap > fit_start_time_idx:
            raise ValueError("Value of fit_start_time_idx must be bigger or equal to window_length + window_gap")

        self.start_time_idx = start_time_idx
        self.stop_time_idx = stop_time_idx
        self.fit_start_time_idx = fit_start_time_idx
        self.fit_stop_time_idx = fit_stop_time_idx


        self.electrodes_name = electrodes_name
        self.electrodes_index = electrodes_name_to_index(electrodes_name)
        self.num_components = 1 # TODO implement multiple components
        self.num_harmonics = num_harmonics

        self.fb_num_subband=fb_num_subband
        self.fb_fundamental_freq = fb_fundamental_freq
        self.fb_upper_bound_freq=fb_upper_bound_freq
        self.fb_weight__a=fb_weight__a
        self.fb_weight__b=fb_weight__b

        self.window_gap=window_gap
        self.window_length=window_length
        self.initialize_pipeline()


    def initialize_pipeline(self):
        self.pipeline = (
            ElectrodesFilter(self.electrodes_index),
            TimeFilter(
                start_time_idx=None,
                stop_time_idx=self.stop_time_idx,
                fit_start_time_idx=None,
                fit_stop_time_idx=self.fit_stop_time_idx
            ),
            SpatioTemporalBank(window_gap=self.window_gap, window_length = self.window_length),
            TimeFilter(
                start_time_idx=self.start_time_idx,
                stop_time_idx=None,
                fit_start_time_idx=self.fit_start_time_idx,
                fit_stop_time_idx=None,
            ),
            FilterbankProjector(
                fb_num_subband=self.fb_num_subband,
                fb_fundamental_freq=self.fb_fundamental_freq,
                fb_upper_bound_freq=self.fb_upper_bound_freq,
            ),
            CCAModeFilter(
                num_components=self.num_components,
                num_harmonics=self.num_harmonics
            ),
            Squeeze(),
            FilterBankPredictProba(
                fb_num_subband=self.fb_num_subband,
                fb_weight__a=self.fb_weight__a,
                fb_weight__b=self.fb_weight__b
            )
        )
