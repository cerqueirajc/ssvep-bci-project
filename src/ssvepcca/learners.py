import scipy
import numpy as np
from sklearn.cross_decomposition import CCA

from ssvepcca.definitions import TARGET_FREQUENCY, NUM_TARGETS, NUM_SAMPLES
from ssvepcca.utils import get_harmonic_columns, electrodes_name_to_index

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


class CCABase():

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


    @staticmethod
    def _check_predict_input(input_array):
        assert input_array.ndim == 2, str(input_array.ndim)
        assert input_array.shape[0] == NUM_SAMPLES, str(input_array.shape[0])


    @staticmethod
    def _check_fit_input(input_array):
        assert input_array.ndim == 4
        assert input_array.shape[-2] == 1500
        assert input_array.shape[1] == NUM_TARGETS


    @staticmethod
    def _filter_eeg(eeg, start_time_index, stop_time_index, electrodes_index):
        eeg_filtered = eeg[..., start_time_index:stop_time_index, :]
        if electrodes_index is not None:
            eeg_filtered = eeg_filtered[..., electrodes_index]
        return eeg_filtered


    def predict_proba(self, eeg):
        self._check_predict_input(eeg)

        eeg = eeg[self.start_time_index:self.stop_time_index, :]
        if self.electrodes_index:
            eeg = eeg[..., self.electrodes_index]

        correlations = np.empty(NUM_TARGETS)

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = get_harmonic_columns(
                freq,
                start_time_index=self.start_time_index,
                stop_time_index=self.stop_time_index,
                num_harmonics=self.num_harmonics
            )
            # increased max_iter, convergence problems
            cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)

            correlations[target] = cca_model.fit_correlation(eeg, harmonic)[0]

        return correlations


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

    def predict(self, eeg):
        probability = self.predict_proba(eeg)
        return probability.argmax(), probability


class CCAFixedCoefficients(CCABase):

    def fit(self, eeg_tensor):

        self._check_fit_input(eeg_tensor)
        eeg_tensor = self._filter_eeg(eeg_tensor, self.start_time_index, self.stop_time_index, self.electrodes_index)
        
        num_blocks = eeg_tensor.shape[0]
        num_electrodes = eeg_tensor.shape[-1]

        self.cca_models = {}

        for target, freq in enumerate(TARGET_FREQUENCY):

            eeg_concatenated = eeg_tensor[:, target, :, :].reshape(-1, num_electrodes)

            harmonic = get_harmonic_columns(
                freq,
                start_time_index=self.start_time_index,
                stop_time_index=self.stop_time_index,
                num_harmonics=self.num_harmonics
            )

            harmonic_concatenated = (
                harmonic[np.newaxis, :, :] # create new dummy dimension that will be expanded
                .repeat(num_blocks, axis=0) # repeate the harmonic matrix for each one of the blocks present in eeg, shape=(num_blocks,time,num_harmonics)
                .reshape(-1, self.num_harmonics * 2) # concatenate harmonics from different blocks in a single lengthy one, shape=(num_blocks*time, num_harmonics)
            )

            cca_model = CCACorrelation(n_components=self.num_components, max_iter=CCA_MAX_ITER, scale=False)
            cca_model.fit(eeg_concatenated, harmonic_concatenated)

            self.cca_models[target] = cca_model

        return self


    def predict_proba(self, eeg):

        self._check_predict_input(eeg)
        eeg = self._filter_eeg(eeg, self.start_time_index, self.stop_time_index, self.electrodes_index)

        correlations = np.empty(NUM_TARGETS)

        for target, freq in enumerate(TARGET_FREQUENCY):
            harmonic = get_harmonic_columns(
                freq,
                start_time_index=self.start_time_index,
                stop_time_index=self.stop_time_index,
                num_harmonics=self.num_harmonics
            )

            correlations[target] = self.cca_models[target].correlation(eeg, harmonic)[0]

        return correlations
    
    def predict(self, eeg):
        probability = self.predict_proba(eeg)
        return probability.argmax(), probability
