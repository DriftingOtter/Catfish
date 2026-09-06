from typing import final

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.sparse import diags, eye as sp_eye
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress

from catfish.core.time_index import TimeIndex
from catfish.core.stdout import print_field, print_heading
from catfish.data.layout import parse_ticker

N:         final = 400
X_SPAN:    final = 3.5


class BenDanielDukeModel:

    def __init__(self, time_index, N=N, x_span=X_SPAN):
        if N < 3:
            raise ValueError("Grid must have at least 3 points.")
        if x_span <= 0.0:
            raise ValueError("x_span must be positive.")
        if not isinstance(time_index, TimeIndex):
            raise ValueError("Invalid time index. Must be one of: Date, Datetime")

        self.data          = pd.DataFrame()
        self.training_data = pd.DataFrame()

        self.x_hist = None
        self.vwap   = None
        self.close  = None

        self.grid  = None
        self.dx    = None
        self.mass  = None
        self.psi   = None

        self.h_eff   = None
        self.gex_eff = None
        self.alpha   = None

        self.forecast_result = None
        self.ticker          = None

        self.time_index = time_index
        self.N          = N
        self.x_span     = x_span

        self._forecast_steps = 150
        self._forecast_dt    = 0.015
        self._forecast_snaps = 50

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

        if data.empty:
            raise ValueError("No rows loaded from file.")

        self.data   = data
        self.ticker = parse_ticker(path)

        return True

    def set_training_period(self, period):
        if self.data.empty:
            raise RuntimeError("Data not loaded.")

        if period is None:
            self.training_data = self.data.copy()
        else:
            # sets latest X-days from the last entry in dataset
            self.training_data = self.data.iloc[-period:].copy()

        if self.training_data.empty:
            raise Exception("No training data found within that period. Please check that 0 <= period <= t_n")

        return True

    def calculate_features(self):
        if self.training_data.empty:
            raise RuntimeError("Training period not set.")

        frame  = self.training_data.replace([np.inf, -np.inf], np.nan)
        valid  = frame[["Close", "Volume"]].notna().all(axis=1)
        frame  = frame.loc[valid]
        close  = frame["Close"].values.astype(float)
        volume = frame["Volume"].fillna(0).values.astype(float)

        self.training_data = frame

        k = min(len(close), len(volume))
        close, volume = close[:k], volume[:k]
        vwap = np.cumsum(close * volume) / np.maximum(np.cumsum(volume), 1e-10)
        x    = close - vwap

        self.h_eff = max(float(np.std(x)), 1e-6)

        if len(x) > 20:
            slope, *_ = linregress(x[:-1], np.diff(x))
            self.gex_eff = float(np.clip(-slope * 100, -3.0, 4.0))
        else:
            self.gex_eff = 1.0

        self.alpha = float(np.clip(self._fit_alpha(x), 0.0, 2.5))

        xl         = max(float(np.max(np.abs(x))) * self.x_span, 4.0 * self.h_eff)
        self.grid  = np.linspace(-xl, xl, self.N)
        self.dx    = float(self.grid[1] - self.grid[0])
        self.mass  = self._build_mass(self.grid)

        self.x_hist = x
        self.vwap   = vwap
        self.close  = close

        if self.x_hist.size == 0:
            raise ValueError("No displacement series calculated.")

        return True

    def _fit_alpha(self, x, n_bins=20):
        lo, hi = np.percentile(x, [5, 95])
        edges  = np.linspace(lo, hi, n_bins + 1)
        cx     = 0.5 * (edges[:-1] + edges[1:])
        dx1    = np.diff(x)
        lv     = np.full(n_bins, np.nan)

        for i in range(n_bins):
            valid = np.where((x[:-1] >= edges[i]) & (x[:-1] < edges[i + 1]))[0]
            if len(valid) > 8:
                lv[i] = float(np.var(dx1[valid])) + 1e-8

        ok = np.isfinite(lv)
        if ok.sum() < 4:
            return 0.30

        A = np.vstack([np.ones(ok.sum()), (cx[ok] / self.h_eff) ** 2]).T
        c = np.linalg.lstsq(A, 1.0 / lv[ok], rcond=None)[0]

        if c[0] <= 0:
            return 0.0
        return float(c[1] / c[0])

    def _build_mass(self, x, alpha=None):
        a = self.alpha if alpha is None else alpha
        return 1.0 + a * (x / self.h_eff) ** 2

    def _hamiltonian(self, gex=None):
        g    = self.gex_eff if gex is None else gex
        m    = self.mass
        h    = self.h_eff
        dx   = self.dx
        Nint = self.N - 2

        m_hp = 0.5 * (m[:-1] + m[1:])
        K    = h ** 2 / (2.0 * dx ** 2)
        V    = 0.5 * g * self.grid ** 2

        diag_m = K * (1.0 / m_hp[1:] + 1.0 / m_hp[:-1]) + V[1:-1]
        off    = -K / m_hp[1:-1]

        return diags([diag_m, off, off], [0, 1, -1], format='csr')

    def _initial_momentum(self, window=15):
        if self.x_hist is None:
            raise RuntimeError("Features not calculated.")

        p0 = np.mean(np.diff(self.x_hist[-window:])) * 4.0
        return float(np.clip(p0, -1.5 * self.h_eff, 1.5 * self.h_eff))

    def init_psi(self, x0, p0=None):
        if self.grid is None:
            raise RuntimeError("Features not calculated.")
        if p0 is None:
            p0 = self._initial_momentum()

        sigma = 0.8 * self.h_eff
        psi   = (np.exp(-(self.grid - x0) ** 2 / (2.0 * sigma ** 2))
                 * np.exp(1j * p0 * self.grid / self.h_eff))
        self.psi = psi / np.sqrt(trapezoid(np.abs(psi) ** 2, self.grid))

        return True

    def _cn_step(self, H_int, dt):
        fac         = 1j * dt / (2.0 * self.h_eff)
        Nint        = self.N - 2
        I_int       = sp_eye(Nint, format='csr')
        psi_int_new = spsolve(
            I_int + fac * H_int,
            (I_int - fac * H_int) @ self.psi[1:-1],
        )
        psi_new       = np.zeros(self.N, dtype=complex)
        psi_new[1:-1] = psi_int_new
        nm            = trapezoid(np.abs(psi_new) ** 2, self.grid)
        self.psi      = psi_new / max(np.sqrt(nm), 1e-30)

    def _obs(self):
        rho   = np.abs(self.psi) ** 2
        xE    = float(trapezoid(self.grid * rho, self.grid))
        x2E   = float(trapezoid(self.grid ** 2 * rho, self.grid))
        sigma = float(np.sqrt(max(x2E - xE ** 2, 0.0)))
        dpsi  = np.gradient(self.psi, self.grid)
        pE    = float(self.h_eff
                      * trapezoid(np.imag(np.conj(self.psi) * dpsi), self.grid))
        J     = (self.h_eff / self.mass) * np.imag(np.conj(self.psi) * dpsi)
        return xE, sigma, pE, rho, J

    def forecast(self, steps=None, dt=None, n_snaps=None):
        if self.psi is None:
            raise RuntimeError("Wavefunction not initialized.")

        steps   = self._forecast_steps if steps is None else steps
        dt      = self._forecast_dt if dt is None else dt
        n_snaps = self._forecast_snaps if n_snaps is None else n_snaps

        H_int = self._hamiltonian()
        xs, ss, ps = [], [], []
        snaps = []
        every = max(1, steps // n_snaps)
        rho = J = None

        for i in range(steps):
            self._cn_step(H_int, dt)
            xE, sigma, pE, rho, J = self._obs()
            xs.append(xE)
            ss.append(sigma)
            ps.append(pE)
            if i % every == 0:
                snaps.append(rho.copy())

        self.forecast_result = dict(
            xE=np.array(xs),
            sigma=np.array(ss),
            pE=np.array(ps),
            snaps=np.array(snaps),
            rho_final=rho,
            J_final=J,
        )

        return True

    def set_forecast_params(self, steps=150, dt=0.015, n_snaps=50):
        if steps < 1:
            raise ValueError("steps must be >= 1.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if n_snaps < 1:
            raise ValueError("n_snaps must be >= 1.")
        self._forecast_steps = steps
        self._forecast_dt    = dt
        self._forecast_snaps = n_snaps

    def get_regime(self):
        if self.gex_eff is None:
            raise RuntimeError("Features not calculated.")
        return "mean-reverting" if self.gex_eff >= 0 else "trending"

    def get_parameters(self):
        if self.gex_eff is None:
            raise RuntimeError("Features not calculated.")
        return {
            "gex_eff": self.gex_eff,
            "alpha":   self.alpha,
            "h_eff":   self.h_eff,
        }

    def get_forecast(self):
        if self.forecast_result is None:
            raise RuntimeError("Forecast not computed.")
        fc = self.forecast_result
        return {
            "xE":    fc["xE"],
            "sigma": fc["sigma"],
            "pE":    fc["pE"],
        }

    def print_results(self):
        if self.forecast_result is None:
            raise RuntimeError("Forecast not computed.")

        fc     = self.forecast_result
        params = self.get_parameters()
        xE     = fc["xE"]
        sig    = fc["sigma"]

        print_heading("BenDaniel–Duke")
        print_field("Ticker", self.ticker or "UNKNOWN")
        print_field("Regime", self.get_regime())
        print_field("GEX_eff", f"{params['gex_eff']:+.3f}")
        print_field("alpha", f"{params['alpha']:.3f}")
        print_field("h_eff", f"{params['h_eff']:.3f}")
        print_field("Forecast horizon", f"{len(xE)} steps")
        print_field("Terminal xE", f"{xE[-1]:+.4f}")
        print_field("Terminal sigma", f"{sig[-1]:.4f}")
