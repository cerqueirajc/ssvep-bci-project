from collections import defaultdict
from abc import ABC, abstractmethod

import scipy
import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import CCA
from typing import Optional

from . import runtime_configuration as rc, NDArrayFloat
from .utils import get_harmonic_columns, electrodes_name_to_index, shift_first_dim



CCA_MAX_ITER = 1000


class CCALearner(CCA):

    def correlation(self, X, Y, n_components=None) -> list[float]:
        if n_components is None or n_components > self.n_components: # type: ignore
            n_components = self.n_components # type: ignore

        x_projection, y_projection = self.transform(X=X, Y=Y)

        return [
            scipy.stats.pearsonr(x_projection[:,n], y_projection[:,n]).statistic
            for n in range(0, n_components)
        ]

    def fit_correlation(self, X, Y, n_components=None):
        return self.fit(X, Y).correlation(X, Y, n_components)


class CCABase(ABC):
    """Canonical Correlation Analysis (CCA) for SSVEP Base Class

    This class implements the abstract class for all CCA-based algorithms for SSVEP target
    identification problems. This class should not be used directly, but inherited with
    final implementations of abstract methods.

    The derived classes should strict to a predefined sequence of operations. These operations
    interface must follow strict rules w.r.t. the `NDArrayFloat` that in and outs the methods.
    
    # **input_eeg**                 (rc.num_samples, rc.num_electrodes)
    # <preprocess>                  (rc.num_samples, rc.num_electrodes)
    # <projection>                  (num_projections, rc.num_samples, rc.num_electrodes)
    # <postprocess>                 (num_projections, rc.num_samples, rc.num_electrodes)
    # <cca>                         (num_targets, num_projections, num_components)
    # <predict_proba>               (num_targets,)
    # <predict>                     (1,)

    """

    def __init__(
        self,
        electrodes_name=None,
        start_time_index=0,
        stop_time_index=1500,
        num_components=1,
        num_harmonics=3,
    ):
        self.start_time_index = start_time_index
        self.stop_time_index = stop_time_index
        self.electrodes_name = electrodes_name
        self.electrodes_index = electrodes_name_to_index(electrodes_name) if electrodes_name else None
        self.num_components = num_components
        self.num_harmonics = num_harmonics
        self.harmonic_column = lambda freq: get_harmonic_columns(
            freq,
            self.start_time_index,
            self.stop_time_index,
            self.num_harmonics
        )


    @staticmethod
    def _check_predict_input(input_array):
        assert input_array.ndim == 3, str(input_array.ndim)



    @staticmethod
    def _check_fit_input(input_array):
        assert input_array.ndim == 4
        assert input_array.shape[-2] == rc.num_samples
        assert input_array.shape[1] == rc.num_targets


    @staticmethod
    def _filter_eeg_electrodes(eeg, electrodes_index):
        if electrodes_index is not None:
            return eeg[..., electrodes_index]
        return eeg


    @staticmethod
    def _filter_eeg_time(eeg, start_time_index: Optional[int], stop_time_index: Optional[int]):
        return eeg[..., start_time_index:stop_time_index, :]


    def get_time_window_size(self):
        return self.stop_time_index - self.start_time_index


    # def feature_extractor(self, eeg: NDArrayFloat) -> NDArrayFloat:
    #     self._check_predict_input(eeg)

    #     eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
    #     eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

    #     correlations = np.empty([rc.num_targets, self.num_components])

    #     for target, freq in enumerate(rc.target_frequencies):
    #         harmonic = self.harmonic_column(freq)
    #         cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
    #         correlations[target, :] = cca_model.fit_correlation(eeg, harmonic)

    #     return correlations

    def preprocess(self, eeg: NDArrayFloat) -> NDArrayFloat:
        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)
        return eeg

    def projection(self, eeg: NDArrayFloat) -> NDArrayFloat:
        return eeg[np.newaxis, :, :]

    def postprocess(self, eeg: NDArrayFloat) -> NDArrayFloat:
        return eeg

    @abstractmethod
    def cca(self, eeg: NDArrayFloat) -> NDArrayFloat:
        pass

    @abstractmethod
    def predict_proba(self, eeg: NDArrayFloat) -> NDArrayFloat:
        pass
    
    @abstractmethod
    def fit(self, eeg_tensor: NDArrayFloat):
        pass

    def predict(self, eeg: NDArrayFloat):
        eeg_preprocessed = self.preprocess(eeg)
        eeg_projected = self.projection(eeg_preprocessed)
        eeg_postprocessed = self.postprocess(eeg_projected)
        eeg_correlations = self.cca(eeg_postprocessed)
        eeg_predict_proba = self.predict_proba(eeg_correlations)
        eeg_predict = eeg_predict_proba.argmax()

        # return eeg_predict, eeg_predict_proba, eeg_correlations
        return eeg_predict, eeg_predict_proba


class CCAModeCorrelationMixin(CCABase):
    def cca(self, eeg):
        self._check_predict_input(eeg)
        
        correlations = np.empty([rc.num_targets, 1, self.num_components])
        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            correlations[target, 0, :] = cca_model.fit_correlation(eeg[0, :, :], harmonic)
             # TODO: Fix this, we need to create one CCA for each projection instead of using always the index 1 proj

        return correlations
    
    def fit(self, eeg_tensor: NDArrayFloat):
        raise NotImplementedError


class CCAModeFilterMixin(CCABase):

    def _check_is_fitted(self) -> None:
        hasattr(self, "cca")

    def cca(self, eeg: NDArrayFloat) -> NDArrayFloat:
        self._check_predict_input(eeg)
        self._check_is_fitted()
        
        correlations = np.empty([rc.num_targets, self.num_components])
        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            correlations[target, :] = self.cca_models[target].correlation(eeg, harmonic)

        return correlations

    def fit(self, eeg_tensor: NDArrayFloat):

        self._check_fit_input(eeg_tensor)
        eeg_tensor = self._filter_eeg_electrodes(eeg_tensor, self.electrodes_index)
        eeg_tensor = self._filter_eeg_time(eeg_tensor, self.start_time_index, self.stop_time_index)

        num_blocks = eeg_tensor.shape[0]
        num_electrodes = eeg_tensor.shape[-1]

        self.cca_models = {}

        for target, freq in enumerate(rc.target_frequencies):

            eeg_concatenated = eeg_tensor[:, target, :, :].reshape(-1, num_electrodes)

            harmonic = self.harmonic_column(freq)

            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(num_blocks * self.get_time_window_size(), self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            cca_model.fit(eeg_concatenated, harmonic_concatenated)

            self.cca_models[target] = cca_model

        return self


class CCAModeFilterGlobalMixin(CCABase):

    def _check_is_fitted(self) -> None:
        hasattr(self, "cca")

    def cca(self, eeg: NDArrayFloat) -> NDArrayFloat:
        self._check_predict_input(eeg)
        self._check_is_fitted()
        
        correlations = np.empty([rc.num_targets, self.num_components])
        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            correlations[target, :] = self.cca_model.correlation(eeg, harmonic)

        return correlations

    def fit(self, eeg_tensor: NDArrayFloat):

        self._check_fit_input(eeg_tensor)
        eeg_tensor = self._filter_eeg_electrodes(eeg_tensor, self.electrodes_index)
        eeg_tensor = self._filter_eeg_time(eeg_tensor, self.start_time_index, self.stop_time_index)

        num_blocks = eeg_tensor.shape[0]
        num_electrodes = eeg_tensor.shape[-1]

        self.cca_models = {}
        """
        Our data is in the format eeg_tensor[blocks, targets, time, electrodes]
        Usually we do everything by completely excluding the target from the equation. We partition everything by target,
        as an individual analysis. However, this limits the ability to learn to distinguish between frequencies. We need
        to learn how to deal with all frequencies simultaneously, not in the vacuum.

        To do so, we will have a single CCA filter, instead of a filter for each target. We need to create a reference
        signal that is coherent, and then reshap everything.

        How to build harmonic reference:
        we need to concatenate the references accordingly
        Harmonic  =  harmonic[blocks, targets, time]
        """

        harmonic = np.zeros((rc.num_targets, self.get_time_window_size()), self.num_harmonics * 2)

        for target, freq in enumerate(rc.target_frequencies):
            harmonic[target, :, :] = self.harmonic_column(freq)

        dim0 = num_blocks * rc.num_targets * self.get_time_window_size()
        
        eeg_concatenated = eeg_tensor.reshape(dim0, num_electrodes)
        
        harmonic_concatenated = (
            harmonic[np.newaxis, :, :, :] # create new dummy dimension that will be expanded
            .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
            .reshape(dim0, self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
        )

        self.cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
        self.cca_model.fit(eeg_concatenated, harmonic_concatenated)

        return self


class CCASingleComponentMixin:
    def predict_proba(self, eeg: NDArrayFloat) -> NDArrayFloat:
        return eeg[:, :, 0]


class CCAMultiComponentMixin:
    def predict_proba(self, eeg: NDArrayFloat) -> NDArrayFloat:
        return eeg.mean(axis=[0, 1]) # this is not a serious implementation, we need to implement the paper's methodology


class CCAStandard(CCASingleComponentMixin, CCAModeCorrelationMixin, CCABase):
    pass

class CCAMultiComponent(CCABase):

    def predict_proba(self, eeg):
        return self.feature_extractor(eeg).mean(axis=1)


class CCAFixedCoefficients(CCASingleComponent):

    def fit(self, eeg_tensor):

        self._check_fit_input(eeg_tensor)
        eeg_tensor = self._filter_eeg_electrodes(eeg_tensor, self.electrodes_index)
        eeg_tensor = self._filter_eeg_time(eeg_tensor, self.start_time_index, self.stop_time_index)

        num_blocks = eeg_tensor.shape[0]
        num_electrodes = eeg_tensor.shape[-1]

        self.cca_models = {}

        for target, freq in enumerate(rc.target_frequencies):

            eeg_concatenated = eeg_tensor[:, target, :, :].reshape(-1, num_electrodes)

            harmonic = self.harmonic_column(freq)

            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(num_blocks * self.get_time_window_size(), self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            cca_model.fit(eeg_concatenated, harmonic_concatenated)

            self.cca_models[target] = cca_model

        return self


    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

        correlations = np.empty([rc.num_targets, self.num_components])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            correlations[target, :] = self.cca_models[target].correlation(eeg, harmonic)

        return correlations


class FilterbankCCA(CCASingleComponent):

    def __init__(
        self,
        electrodes_name=None,
        start_time_index=0,
        stop_time_index=1500,
        # num_components=1,
        num_harmonics=3,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
    ):
        super().__init__(electrodes_name, start_time_index, stop_time_index, num_harmonics)

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
            assert freq_low < freq_high, f"Number of FB subband is too high, try decreasing it"

            b, a = scipy.signal.cheby1(
                Wn=[freq_low - self.FB_BAND_FREQ_TOL, freq_high + self.FB_BAND_FREQ_TOL],
                N=self.CHEBY_ORDER,
                rp=self.CHEBY_MAX_RIPPLE,
                btype=self.CHEBY_BAND_TYPE,
                fs=self.CHEBY_FS
            )

            self.fb_filters.append({"b": b, "a": a})

        def fb_weight(n):
            a = self.fb_weight__a
            b = self.fb_weight__b
            return n**(-a) + b

        self.fb_weight_array = np.array([fb_weight(n) for n in range(1, self.fb_num_subband + 1)])


    def apply_filter_bank(self, eeg):
        # expand axis to add filtered dimensions
        eeg_fb = (
            eeg[..., np.newaxis, :, :]
            .repeat(self.fb_num_subband, axis=-3)
        )
        # filter
        for filter_idx, filter_params in enumerate(self.fb_filters):
           for feature_idx in range(eeg_fb.shape[-1]):
                eeg_fb[..., filter_idx, :, feature_idx] = scipy.signal.filtfilt(
                    x=eeg_fb[filter_idx, :, feature_idx],
                    b=filter_params["b"],
                    a=filter_params["a"],
                    padtype=self.FB_PADDING_TYPE
                )

        return eeg_fb


    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

        eeg_fb = self.apply_filter_bank(eeg)

        correlations = np.empty([rc.num_targets, self.fb_num_subband])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            for filter_idx in range(self.fb_num_subband):
                cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
                correlations[target, filter_idx] = cca_model.fit_correlation(eeg_fb[filter_idx, ...], harmonic)[0]

        return correlations


    def predict_proba(self, eeg):
        features = self.feature_extractor(eeg)
        return np.power(features, 2) @ self.fb_weight_array


class FBCCAFixedCoefficients(FilterbankCCA):

    def fit(self, eeg_tensor):

        self._check_fit_input(eeg_tensor)
        eeg_tensor = self._filter_eeg_electrodes(eeg_tensor, self.electrodes_index)
        eeg_tensor = self._filter_eeg_time(eeg_tensor, self.start_time_index, self.stop_time_index)

        num_blocks = eeg_tensor.shape[0]

        self.cca_models = defaultdict(lambda : {})

        for target, freq in enumerate(rc.target_frequencies):

            eeg_fb = self.apply_filter_bank(eeg_tensor[:, target, :, :])

            harmonic = self.harmonic_column(freq)
            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(-1, self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            for filter_idx in range(self.fb_num_subband):
                cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
                cca_model.fit(eeg_fb[filter_idx, ...], harmonic_concatenated)
                self.cca_models[target][filter_idx] = cca_model

        return self


    def apply_filter_bank(self, eeg):
        eeg_shape = eeg.shape # dim = (blocks, time_window, electrodes)
        eeg_transposed = eeg.transpose([1, 0, 2]).reshape(eeg_shape[1], -1)
        eeg_fb_transposed = super().apply_filter_bank(eeg_transposed)
        eeg_fb = (
            eeg_fb_transposed # dim = (num_subbands, time_window, blocks * electrodes)
            .reshape(self.fb_num_subband, eeg_shape[1], eeg_shape[0], eeg_shape[2])
            .transpose([0, 2, 1, 3])
            .reshape(self.fb_num_subband, -1, eeg_shape[2])
        )
        return eeg_fb


    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

        eeg_fb = super().apply_filter_bank(eeg)

        correlations = np.empty([rc.num_targets, self.fb_num_subband])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            for filter_idx in range(self.fb_num_subband):
                cca_model = self.cca_models[target][filter_idx]
                correlations[target, filter_idx] = cca_model.correlation(eeg_fb[filter_idx, ...], harmonic)[0]

        return correlations


class AlternativeFBCCA(FilterbankCCA):

    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

        eeg_fb = self.apply_filter_bank(eeg)
        eeg_fb = np.transpose(eeg_fb, axes=(1,2,0)) # time, electrodes, fb_number
        eeg_fb = eeg_fb.reshape(self.get_time_window_size(), -1) # use filterbank as new features

        correlations = np.empty([rc.num_targets])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)

            cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            correlations[target] = cca_model.fit_correlation(eeg_fb, harmonic)[0]

        return correlations


    def predict_proba(self, eeg):
        return self.feature_extractor(eeg)


class CCASpatioTemporal(CCASingleComponent):

    def __init__(
        self,
        electrodes_name=None,
        start_time_index=0,
        stop_time_index=1500,
        num_harmonics=3,
        window_gap=0,
        window_length=10
    ):
        self.window_gap = window_gap
        self.window_length = window_length

        assert self.window_gap >= 0
        assert self.window_length >= 0

        super().__init__(electrodes_name, start_time_index, stop_time_index, num_harmonics)


    def apply_fir_bank(self, eeg):

        eeg_with_lags =  (
            eeg[..., np.newaxis]
            .repeat(self.window_length + 1, axis=-1)
        )

        for length in range(1, self.window_length + 1): # skip zero, always keep t=0 as original value
            lag = length + self.window_gap
            eeg_with_lags[..., length] = shift_first_dim(eeg_with_lags[..., length], lag)

        return eeg_with_lags


    def preprocess_fir_eeg(self, eeg):

        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, None, self.stop_time_index)

        eeg_with_lags = self.apply_fir_bank(eeg)
        eeg_with_lags = eeg_with_lags.reshape(eeg_with_lags.shape[:-2] + (-1,))
        eeg_with_lags = self._filter_eeg_time(eeg_with_lags, self.start_time_index, None)

        return eeg_with_lags


    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg_with_lags = self.preprocess_fir_eeg(eeg)

        correlations = np.empty([rc.num_targets, self.num_components])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            correlations[target, :] = cca_model.fit_correlation(eeg_with_lags, harmonic)

        return correlations


class CCASpatioTemporalFixed(CCASpatioTemporal):

    def fit(self, eeg_tensor):

        self._check_fit_input(eeg_tensor)
        num_blocks = eeg_tensor.shape[0]

        eeg_tensor_with_lags = np.stack(
            [
                self.preprocess_fir_eeg(eeg_tensor[block, target, ...])
                for block in range(eeg_tensor.shape[0])
                for target in range(eeg_tensor.shape[1])
            ]
        ).reshape(eeg_tensor.shape[0], eeg_tensor.shape[1], self.get_time_window_size(), -1)

        self.cca_models = {}

        for target, freq in enumerate(rc.target_frequencies):

            eeg_concatenated = eeg_tensor_with_lags[:, target, :, :].reshape(
                eeg_tensor_with_lags.shape[0] * eeg_tensor_with_lags.shape[2], -1
            )

            harmonic = self.harmonic_column(freq)
            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(-1, self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            cca_model.fit(eeg_concatenated, harmonic_concatenated)

            self.cca_models[target] = cca_model

        return self

    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg_with_lags = self.preprocess_fir_eeg(eeg)

        correlations = np.empty([rc.num_targets, self.num_components])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            correlations[target, :] = self.cca_models[target].correlation(eeg_with_lags, harmonic)

        return correlations


class FBSpatioTemporalCCA(FilterbankCCA, CCASpatioTemporal):

    def __init__(
        self,
        electrodes_name=None,
        start_time_index=0,
        stop_time_index=1500,
        # num_components=1,
        num_harmonics=3,
        window_gap=0,
        window_length=10,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
    ):

        FilterbankCCA.__init__(
            self,
            electrodes_name=electrodes_name,
            start_time_index=start_time_index,
            stop_time_index=stop_time_index,
            # num_components,
            num_harmonics=num_harmonics,
            fb_num_subband=fb_num_subband,
            fb_fundamental_freq=fb_fundamental_freq,
            fb_upper_bound_freq=fb_upper_bound_freq,
            fb_weight__a=fb_weight__a,
            fb_weight__b=fb_weight__b,
        )

        CCASpatioTemporal.__init__(
            self,
            electrodes_name=electrodes_name,
            start_time_index=start_time_index,
            stop_time_index=stop_time_index,
            num_harmonics=num_harmonics,
            window_gap=window_gap,
            window_length=window_length
        )

    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)

        eeg_with_lags = self.preprocess_fir_eeg(eeg)
        eeg_fb = self.apply_filter_bank(eeg_with_lags)

        correlations = np.empty([rc.num_targets, self.fb_num_subband, self.num_components])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            for filter_idx in range(self.fb_num_subband):
                cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
                correlations[target, filter_idx, :] = cca_model.fit_correlation(eeg_fb[filter_idx, ...], harmonic)

        return correlations[:, :, 0]


class FBSpatioTemporalCCAFixed(FBSpatioTemporalCCA):

    def fit(self, eeg_tensor):

        self._check_fit_input(eeg_tensor)
        num_blocks = eeg_tensor.shape[0]
        num_targets = eeg_tensor.shape[1]

        eeg_tensor_preprocessed = np.stack(
            [
                self.apply_filter_bank(
                    self.preprocess_fir_eeg(
                        eeg_tensor[block, target, ...]
                    )
                )
                for block in range(eeg_tensor.shape[0])
                for target in range(eeg_tensor.shape[1])
            ]
        )
        eeg_tensor_preprocessed = eeg_tensor_preprocessed.reshape(
            (num_blocks, num_targets) + eeg_tensor_preprocessed.shape[1:]
        )

        eeg_tensor_preprocessed = eeg_tensor_preprocessed.transpose([1, 2, 0, 3, 4])
        eeg_tensor_preprocessed = eeg_tensor_preprocessed.reshape(
            eeg_tensor_preprocessed.shape[:-3] + (-1, eeg_tensor_preprocessed.shape[-1])
        )

        self.cca_models = defaultdict(lambda : {})

        for target, freq in enumerate(rc.target_frequencies):

            harmonic = self.harmonic_column(freq)
            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(-1, self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            for filter_idx in range(self.fb_num_subband):
                cca_model = CCALearner(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
                cca_model.fit(eeg_tensor_preprocessed[target, filter_idx, ...], harmonic_concatenated)
                self.cca_models[target][filter_idx] = cca_model

        return self


    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)

        eeg_with_lags = self.preprocess_fir_eeg(eeg)
        eeg_fb = self.apply_filter_bank(eeg_with_lags)

        correlations = np.empty([rc.num_targets, self.fb_num_subband, self.num_components])

        for target, freq in enumerate(rc.target_frequencies):
            harmonic = self.harmonic_column(freq)
            for filter_idx in range(self.fb_num_subband):
                cca_model = self.cca_models[target][filter_idx]
                correlations[target, filter_idx, :] = cca_model.correlation(eeg_fb[filter_idx, ...], harmonic)

        return correlations[:, :, 0]
