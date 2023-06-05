import scipy
import numpy as np
from sklearn.cross_decomposition import CCA

from definitions import TARGET_FREQUENCY, NUM_TARGETS
from utils import get_harmonic_columns, electrodes_name_to_index


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

    def predict_proba(self, eeg):

        if self.electrodes_index:
            eeg = eeg[self.start_time_index:self.stop_time_index, self.electrodes_index]
        else:
            eeg = eeg[self.start_time_index:self.stop_time_index, :]

        correlations = np.empty(NUM_TARGETS)

        for idx, freq in enumerate(TARGET_FREQUENCY):
            harmonic = get_harmonic_columns(
                freq,
                start_time_index=self.start_time_index,
                stop_time_index=self.stop_time_index,
                num_harmonics=self.num_harmonics
            )
            # increased max_iter, convergence problems
            cca_model = CCACorrelation(n_components=self.num_components, max_iter=1000, scale=False)

            correlations[idx] = cca_model.fit_correlation(eeg, harmonic)[0]

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
        correlations = self.predict_proba(eeg)
        return correlations.argmax()
