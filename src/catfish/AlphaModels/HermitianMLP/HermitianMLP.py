from typing import final

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from catfish.AlphaModels.TimeIndex import TimeIndex

WINDOW:     final = 2 * 252
MIN_WINDOW: final = 60
TAU:        final = 0.52


class HermitianMLPModel:

    def __init__(self, time_index):
        self.data     = pd.DataFrame()
        self.state_df = pd.DataFrame()
        self.results  = pd.DataFrame()

        self.model  = None
        self.scaler = None

        self.time_index  = time_index
        self.feat_cols   = None
        self.tau         = TAU
        self.window      = WINDOW
        self.scaler_type = StandardScaler()

        if not isinstance(time_index, TimeIndex):
            raise ValueError("Invalid time index. Must be one of: Date, Datetime")

    def _log_return_(self, data):
        ratio = data['Close'] / data['Close'].shift(1)
        data['r'] = np.where(ratio > 0.0, np.log(ratio), np.nan)
        return data

    def _volatility_(self, data):
        data['sigma'] = data['r'].rolling(20).std()
        return data

    def _signed_vol_(self, data):
        price_range = data['High'] - data['Low']
        imbalance   = np.where(
            price_range > 0.0,
            (2 * data['Open'] - data['High'] - data['Low']) / price_range,
            0.0,
        )
        vol = np.log1p(np.where(data['Volume'] > 0.0, data['Volume'], 0.0))

        # avoid (price_range <= 0.0) -> divide by zero error
        # avoid (Volume <= 0.0) -> zero-volume bars
        data['phi'] = np.where(
            (price_range > 0.0) & (data['Volume'] > 0.0),
            imbalance * vol,
            0.0,
        )
        return data

    def _displacement_(self, data):
        mu = data['Close'].rolling(20).mean()
        s  = data['Close'].rolling(20).std()
        data['D'] = np.where(s > 0.0, (data['Close'] - mu) / s, 0.0)
        return data

    def _rsi_(self, data):
        delta    = data['Close'].diff()
        avg_gain = delta.clip(lower=0).rolling(14).mean()
        avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
        data['rsi'] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-12))
        return data


    @staticmethod
    def _hermite_coeffs(x):
        # mean-subtract only — preserves variance information in c2
        z    = x - x.mean()
        c2   = np.mean(z ** 2 - 1)
        c3   = np.mean(z ** 3 - 3 * z)
        # signed second moment — identifies which tail drives the variance
        c_pm = np.mean(np.sign(z) * z ** 2)
        return c2, c3, c_pm

    def _embed(self):
        features    = ['r', 'sigma', 'phi', 'D', 'rsi']
        rows, dates = [], []

        for i in range(self.window, len(self.data)):
            row = {}
            for f in features:
                x            = self.data[f].iloc[i - self.window:i].values
                c2, c3, c_pm = self._hermite_coeffs(x)
                row[f'c2_{f}']  = c2
                row[f'c3_{f}']  = c3
                row[f'cpm_{f}'] = c_pm
            rows.append(row)
            dates.append(self.data.index[i])

        state         = pd.DataFrame(rows)
        state['date'] = dates

        state['next_r'] = self.data['r'].shift(-1).reindex(state['date']).values
        state['next_d'] = (state['next_r'] > 0).astype(int)

        return state.dropna()

    def load_data(self, path):
        if not path.endswith('.csv'):
            raise Exception('Invalid file format. Must be .csv')

        data = pd.read_csv(path)

        if self.time_index == TimeIndex.Date:
            col = "Date"
        else:
            col = "Datetime"

        if col not in data.columns:
            raise Exception(f'Expected column "{col}" for {self.time_index.name} data.')

        data[col] = pd.to_datetime(data[col])
        data = data.sort_values(by=[col])
        data = data.set_index(col)
        data = data[~data.index.duplicated(keep="last")]

        self.data = data

        return True

    def calculate_features(self):
        if self.data.empty:
            raise RuntimeError('Data not loaded.')

        data = self.data.copy()
        data = self._log_return_(data)
        data = self._volatility_(data)
        data = self._signed_vol_(data)
        data = self._displacement_(data)
        data = self._rsi_(data)
        self.data = data.replace([np.inf, -np.inf], np.nan).dropna()
        n             = len(self.data)
        self.window   = min(WINDOW, n // 2)
        if self.window < MIN_WINDOW:
            raise ValueError(
                f'Not enough rows after feature prep ({n}); '
                f'need at least {2 * MIN_WINDOW} for embedding.'
            )

        self.state_df = self._embed()

        exclude        = {'date', 'next_r', 'next_d'}
        self.feat_cols = [c for c in self.state_df.columns if c not in exclude]

        if self.state_df.empty:
            raise ValueError(
                f'State embedding is empty ({n} rows, window={self.window}).'
            )

        return True

    def init_model(self):
        if self.state_df.empty:
            raise RuntimeError('Features not calculated.')

        self.scaler = StandardScaler()

        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=0,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

        return True

    def train_model(self):
        if self.model is None or self.scaler is None:
            raise RuntimeError('Model not initialized.')
        train = self.state_df
        if train.empty:
            raise ValueError('Training set is empty.')

        X_tr = self.scaler.fit_transform(train[self.feat_cols])
        y_tr = train['next_d'].values
        self.model.fit(X_tr, y_tr)

        if self.model.n_iter_ >= self.model.max_iter:
            return False

        return True

    def generate_signals(self):
        if self.model is None:
            raise RuntimeError('Model not trained.')
        test = self.state_df.copy()
        if test.empty:
            raise ValueError('No state rows to score.')

        X_te               = self.scaler.transform(test[self.feat_cols])
        test['mlp_proba']  = self.model.predict_proba(X_te)[:, 1]
        test['mlp_signal'] = (test['mlp_proba'] > self.tau).astype(int)
        self.results       = test.reset_index(drop=True)

        return True


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from catfish.AlphaModels.HermitianMLP import HermitianMLPViz as vz
    from catfish.paths import PROJECT_ROOT

    Model = HermitianMLPModel(time_index=TimeIndex.Datetime)
    Model.load_data(str(PROJECT_ROOT / "datasets" / "QQQ" / "QQQ-2026-06-11.csv"))

    #Model = HermitianMLPModel(time_index=TimeIndex.Date)
    #Model.load_data(str(PROJECT_ROOT / "datasets" / "QQQ" / "QQQ.csv"))
    Model.calculate_features()
    Model.init_model()

    _ = Model.train_model()
    if _ is False:
        raise Exception('MLP did not converge within iteration limit.')

    Model.generate_signals()

    Viz = vz.HermitianViz(Model)
    Viz.plot_all()
    plt.show()