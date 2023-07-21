from collections import defaultdict
import scipy
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy import signal

from ssvepcca.definitions import TARGET_FREQUENCY, NUM_TARGETS, NUM_SAMPLES, SAMPLE_FREQ
from ssvepcca.utils import get_harmonic_columns, electrodes_name_to_index, shift_first_dim


CCA_MAX_ITER = 1000


class CCACorrelation(CCA):

    def correlation(self, X, Y, n_components=None):
        if n_components is None or n_components > self.n_components:
            n_components = self.n_components

        x_projection, y_projection = self.transform(X=X, Y=Y)

        return [
            scipy.stats.pearsonr(x_projection[:,n], y_projection[:,n]).statistic
            for n in range(0, n_components)
        ]

    def fit_correlation(self, X, Y, n_components=None):
        return self.fit(X, Y).correlation(X, Y, n_components)


class CCABase:

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
        assert input_array.ndim == 2, str(input_array.ndim)
        assert input_array.shape[0] == NUM_SAMPLES, str(input_array.shape[0])


    @staticmethod
    def _check_fit_input(input_array):
        assert input_array.ndim == 4
        assert input_array.shape[-2] == NUM_SAMPLES
        assert input_array.shape[1] == NUM_TARGETS


    @staticmethod
    def _filter_eeg_electrodes(eeg, electrodes_index):
        if electrodes_index is not None:
            return eeg[..., electrodes_index]
        return eeg


    @staticmethod
    def _filter_eeg_time(eeg, start_time_index, stop_time_index):
        return eeg[..., start_time_index:stop_time_index, :]


    def feature_extractor(self, eeg):
        self._check_predict_input(eeg)

        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

        correlations = np.empty([NUM_TARGETS, self.num_components])

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = self.harmonic_column(freq)
            cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            correlations[target, :] = cca_model.fit_correlation(eeg, harmonic)

        return correlations


    def predict_proba(self, eeg):
        raise NotImplementedError


    def predict(self, eeg):
        probability = self.predict_proba(eeg)
        return probability.argmax(), probability


class CCASingleComponent(CCABase):

    def __init__(
        self,
        electrodes_name=None,
        start_time_index=0,
        stop_time_index=1500,
        num_harmonics=3,
    ):
        return super().__init__(
            electrodes_name=electrodes_name,
            start_time_index=start_time_index,
            stop_time_index=stop_time_index,
            num_components=1,
            num_harmonics=num_harmonics,
        )

    def predict_proba(self, eeg):
        return self.feature_extractor(eeg)[:, 0]


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

        for target, freq in enumerate(TARGET_FREQUENCY):

            eeg_concatenated = eeg_tensor[:, target, :, :].reshape(-1, num_electrodes)

            harmonic = self.harmonic_column(freq)

            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(-1, self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            cca_model.fit(eeg_concatenated, harmonic_concatenated)

            self.cca_models[target] = cca_model

        return self


    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

        correlations = np.empty([NUM_TARGETS, self.num_components])

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = self.harmonic_column(freq)
            correlations[target, :] = self.cca_models[target].correlation(eeg, harmonic)

        return correlations


class FBCCA(CCABase):

    CHEBY_ORDER = 6
    CHEBY_MAX_RIPPLE = 0.5
    CHEBY_FS = SAMPLE_FREQ
    CHEBY_BAND_TYPE = "bandpass"
    FB_BAND_FREQ_TOL = 2
    FB_PADDING_TYPE = "odd"

    def __init__(
        self,
        electrodes_name=None,
        start_time_index=0,
        stop_time_index=1500,
        num_components=1,
        num_harmonics=3,
        fb_num_subband=3,
        fb_fundamental_freq=8,
        fb_upper_bound_freq=88,
        fb_weight__a=1.25,
        fb_weight__b=0.25,
    ):
        super().__init__(electrodes_name, start_time_index, stop_time_index, num_components, num_harmonics)

        self.fb_num_subband=fb_num_subband
        self.fb_fundamental_freq = fb_fundamental_freq
        self.fb_upper_bound_freq=fb_upper_bound_freq
        self.fb_weight__a=fb_weight__a
        self.fb_weight__b=fb_weight__b

        self.fb_filters = []
        for i in range(self.fb_num_subband):

            freq_low = fb_fundamental_freq * (i + 1)
            freq_high = self.fb_upper_bound_freq
            assert freq_low < freq_high, f"Number of FB subband is too high, try decreasing it"

            b, a = signal.cheby1(
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
            eeg[np.newaxis, :, :]
            .repeat(self.fb_num_subband, axis=0)
        )
        # filter
        for filter_idx, filter_params in enumerate(self.fb_filters):
           for feature_idx in range(eeg_fb.shape[-1]):
                eeg_fb[filter_idx, :, feature_idx] = signal.filtfilt(
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

        correlations = np.empty([NUM_TARGETS, self.fb_num_subband])

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = self.harmonic_column(freq)
            for filter_idx in range(self.fb_num_subband):
                cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
                correlations[target, filter_idx] = cca_model.fit_correlation(eeg_fb[filter_idx, ...], harmonic)[0]

        return correlations


    def predict_proba(self, eeg):
        features = self.feature_extractor(eeg)
        return np.power(features, 2) @ self.fb_weight_array


class FBCCAFixedCoefficients(FBCCA):

    def fit(self, eeg_tensor):

        self._check_fit_input(eeg_tensor)
        eeg_tensor = self._filter_eeg_electrodes(eeg_tensor, self.electrodes_index)
        eeg_tensor = self._filter_eeg_time(eeg_tensor, self.start_time_index, self.stop_time_index)

        num_blocks = eeg_tensor.shape[0]

        self.cca_models = defaultdict(lambda : {})

        for target, freq in enumerate(TARGET_FREQUENCY):

            eeg_fb = self.apply_filter_bank(eeg_tensor[:, target, :, :])

            harmonic = self.harmonic_column(freq)
            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(-1, self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            for filter_idx in range(self.fb_num_subband):
                cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
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

        correlations = np.empty([NUM_TARGETS, self.fb_num_subband])

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = self.harmonic_column(freq)
            for filter_idx in range(self.fb_num_subband):
                cca_model = self.cca_models[target][filter_idx]
                correlations[target, filter_idx] = cca_model.correlation(eeg_fb[filter_idx, ...], harmonic)[0]

        return correlations


class AlternativeFBCCA(FBCCA):

    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)
        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, self.start_time_index, self.stop_time_index)

        eeg_fb = self.apply_filter_bank(eeg)
        eeg_fb = np.transpose(eeg_fb, axes=(1,2,0)) # time, electrodes, fb_number
        eeg_fb = eeg_fb.reshape(self.stop_time_index - self.start_time_index, -1) # use filterbank as new features

        correlations = np.empty([NUM_TARGETS])

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = self.harmonic_column(freq)
            
            cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
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
        window_step=1,
        window_length=10
    ):
        self.window_step = window_step
        self.window_length = window_length

        assert self.window_step > 0
        assert self.window_length > 0

        super().__init__(electrodes_name, start_time_index, stop_time_index, num_harmonics)

    def feature_extractor(self, eeg):

        self._check_predict_input(eeg)

        eeg = self._filter_eeg_electrodes(eeg, self.electrodes_index)
        eeg = self._filter_eeg_time(eeg, None, self.stop_time_index)

        eeg_with_lags =  (
            eeg[np.newaxis, :, :]
            .repeat(self.window_length, axis=0)
        )

        for length in range(self.window_length):
            lag = length + self.window_step
            eeg_with_lags[length, ...] = shift_first_dim(eeg_with_lags[length, ...], lag)

        eeg_with_lags = self._filter_eeg_time(eeg_with_lags, self.start_time_index, None)
        eeg_with_lags = np.transpose(eeg_with_lags, (1,2,0)).reshape(self.stop_time_index - self.start_time_index, -1)

        correlations = np.empty([NUM_TARGETS, self.num_components])

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = self.harmonic_column(freq)
            cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            correlations[target, :] = cca_model.fit_correlation(eeg_with_lags, harmonic)

        return correlations
