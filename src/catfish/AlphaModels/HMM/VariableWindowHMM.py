import enum
from typing import final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.preprocessing import RobustScaler, StandardScaler

from catfish.AlphaModels.HMM import VariableWindowHMMViz as vz

ALPHA: final = 4
GAMMA: final = 6
DELTA: final = 2


class ModelType(enum.Enum):
    GaussianEmission  = 0
    GaussianMixture   = 1


class MarketPressureModel:

    def __init__(self, model_type):
        self.data           = pd.DataFrame()
        self.training_data  = pd.DataFrame()
        self.feature_vector = pd.DataFrame()

        self.model  = None
        self.scaler = None

        self.model_type = model_type
        if not isinstance(model_type, ModelType):
            raise ValueError("Invalid model type. Must be one of: GaussianHMM, GaussianMixture")

    def load_data(self, path):
        if not path.endswith('.csv'):
            raise Exception('Invalid file format. Must be .csv')

        data = pd.read_csv(path)

        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values(by=["Date"])
        data = data.set_index("Date")

        self.data = data

    def set_training_period(self, period):
        if period is None:
            self.training_data = self.data.copy()
        else:
            # sets latest X-days from the last entry in dataset
            self.training_data = self.data.iloc[-period:].copy()

        if self.training_data.empty:
            raise Exception("No training data found within that period. Please check that 0 <= period <= t_n")

        return True

    def _log_return_(self, data):
        data["r"] = np.log(data["Close"] / data["Close"].shift(1))
        return data

    def _signed_vol_proxy_(self, data):
        price_range = data["High"] - data["Low"]
        imbalance = (2.0 * data["Close"] - data["High"] - data["Low"]) / price_range

        # avoid (price_range <= 0.0) -> divide by zero error
        data["phi"] = np.where(
            price_range > 0.0,
            imbalance * np.log(data["Volume"]),
            0.0,
        )

        if data["phi"].isna().any():
            raise ValueError("phi has null values.")

        return data

    def calculate_features(self):
        self._log_return_(self.training_data)
        self._signed_vol_proxy_(self.training_data)

        features = self.training_data[["r", "phi"]].dropna()
        self.training_data = self.training_data.loc[features.index]

        if self.model_type == ModelType.GaussianEmission:
            self.scaler = RobustScaler()
        if self.model_type == ModelType.GaussianMixture:
            self.scaler = StandardScaler()
        if self.scaler is None:
            raise Exception('Invalid model type. Must be one of: GaussianHMM, GaussianMixture')

        self.feature_vector = self.scaler.fit_transform(features.to_numpy())

        if self.feature_vector.size == 0:
            raise Exception("No features vector calculated.")

        return True

    def init_model(self):
        if self.model_type is None:
            raise Exception('Invalid model type. Must be one of: GaussianHMM, GaussianMixture')

        if self.model_type == ModelType.GaussianEmission:
            self.model = GaussianHMM(
                n_components=ALPHA,
                covariance_type="full",
                n_iter=5000,
                tol=1e-4,
                random_state=42
            )

        if self.model_type == ModelType.GaussianMixture:
            self.model = GMMHMM(
                n_components=GAMMA,
                n_mix=DELTA,
                covariance_type="full",
                n_iter=2000,
                tol=1e-4,
                random_state=42
            )

    def train_model(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.fit(self.feature_vector)

        if not self.model.monitor_.converged:
            return False

        return True

    def predict_current_state(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        states = self.model.predict(self.feature_vector)
        return states[-1]

    def predict_state_probabilities(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        probs = self.model.predict_proba(self.feature_vector)
        return probs[-1]

    def get_state_summary(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        states = self.model.predict(self.feature_vector)

        data = self.training_data.copy()
        data["state"] = states

        summary = data.groupby("state").agg({
            "r": ["mean", "std", "count"],
            "phi": ["mean", "std"]
        })

        return summary

    def get_transition_matrix(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        return self.model.transmat_

    def get_current_regime_distribution(self):
        probs = self.model.predict_proba(self.feature_vector)
        return probs[-1]

    def get_state_path(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        return self.model.predict(self.feature_vector)

    def get_regime_volatility(self):
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        states = self.model.predict(self.feature_vector)

        df = self.training_data.iloc[1:].copy()
        df["state"] = states

        return df.groupby("state")["r"].std()

    def _state_emission_means(self):
        if self.model.model_type == ModelType.GaussianEmission:
            return self.model.model.means_
        weights = self.model.model.weights_
        means = self.model.model.means_
        return np.einsum('sm,smf->sf', weights, means)

    def _state_emission_variances(self):
        if self.model.model_type == ModelType.GaussianEmission:
            return np.array([np.diag(self.model.model.covars_[s])
                             for s in range(self.model.model.n_components)])

        weights = self.model.model.weights_
        means = self.model.model.means_
        state_means = np.einsum('sm,smf->sf', weights, means)
        n_states = self.model.model.n_components
        variances = np.zeros_like(state_means)

        for s in range(n_states):
            for m in range(self.model.model.n_mix):
                w = weights[s, m]
                mu_m = means[s, m]
                sig_m = np.diag(self.model.model.covars_[s, m])
                variances[s] += w * (sig_m + (mu_m - state_means[s]) ** 2)

        return variances

    def _forward_project(self, n_ahead):
        transmat = self.model.get_transition_matrix()
        current_probs = self.model.get_current_regime_distribution()
        state_means = self._state_emission_means(self.model)
        state_vars = self._state_emission_variances(self.model)
        n_states = self.model.model.n_components

        state_probs = np.zeros((n_ahead, n_states))
        probs = current_probs.copy()

        for t in range(n_ahead):
            probs = probs @ transmat
            state_probs[t] = probs

        # expected emission and total variance (law of total variance)
        expected = state_probs @ state_means
        variance = np.zeros_like(expected)

        for s in range(n_states):
            p_s = state_probs[:, s:s + 1]
            variance += p_s * (state_vars[s] + (state_means[s] - expected) ** 2)

        std_original = np.sqrt(variance) * self.model.scaler.scale_
        expected_original = self.model.scaler.inverse_transform(expected)

        return state_probs, expected_original, std_original


if __name__ == '__main__':

    from catfish.paths import PROJECT_ROOT

    ShortModel = MarketPressureModel(model_type=ModelType.GaussianEmission)
    ShortModel.load_data(str(PROJECT_ROOT / "datasets" / "QQQ" / "QQQ.csv"))

    ShortModel.set_training_period(period=252)
    ShortModel.calculate_features()
    ShortModel.init_model()

    _ = ShortModel.train_model()
    if _ is False:
        raise Exception("Convergence failed")

    Viz = vz.Plotter(ShortModel)
    fig = Viz.plot_all()
    plt.show()



    #LongModel = MarketPressureModel(model_type=ModelType.GaussianMixture)
    #LongModel.load_data("../../datasets/QQQ.csv")

    #LongModel.set_training_period(period=None)
    #LongModel.calculate_features()
    #LongModel.init_model()

    #_ = LongModel.train_model()
    #if _ is False:
    #    raise Exception("Convergence failed")
