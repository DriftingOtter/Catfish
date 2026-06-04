from typing import final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS: final = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#F44336']


class HermitianViz:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _format_date_axis(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    @staticmethod
    def _future_trading_dates(last_date, n_ahead):
        return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_ahead)

    @staticmethod
    def _make_ax(fig, col_x, col_w, row_y, row_h, row, v_gap, rowspan=1, projection=None):
        last = row + rowspan - 1
        b    = row_y[last]
        h    = sum(row_h[row:last + 1]) + (rowspan - 1) * v_gap
        return fig.add_axes([col_x, b, col_w, h], projection=projection)

    def _simulate_paths(self, n_ahead, n_paths=200, seed=0):
        rng    = np.random.default_rng(seed)
        recent = self.model.data.iloc[-63:]

        r_mean   = recent['r'].mean()
        r_std    = recent['r'].std()
        phi_mean = recent['phi'].mean()
        phi_std  = recent['phi'].std()
        rho      = recent[['r', 'phi']].corr().iloc[0, 1]

        if np.isnan(rho):
            rho = 0.0

        # correlated bivariate normal paths from recent 63-period empirical distribution
        cov   = np.array([[r_std ** 2,              rho * r_std * phi_std],
                          [rho * r_std * phi_std,   phi_std ** 2         ]])
        mu    = np.array([r_mean, phi_mean])
        paths = rng.multivariate_normal(mu, cov, size=(n_paths, n_ahead))

        return paths[:, :, 0], paths[:, :, 1]

    def plot_features(self, axes=None, n_ahead=20, n_paths=200):
        data  = self.model.data
        state = self.model.state_df.set_index('date')
        dates = state.index

        r_vals   = data['r'].reindex(dates).values
        phi_vals = data['phi'].reindex(dates).values
        pos_r    = state['c3_r'].values   > 0
        pos_phi  = state['c3_phi'].values > 0

        last_date    = data.index[-1]
        future_dates = self._future_trading_dates(last_date, n_ahead)
        r_paths, phi_paths = self._simulate_paths(n_ahead, n_paths)

        # anchor forecast to last historical point for visual continuity
        anchor_dates = pd.DatetimeIndex([last_date]).append(future_dates)
        last_r       = float(data['r'].iloc[-1])
        last_phi     = float(data['phi'].iloc[-1])

        r_q05, r_q25, r_med, r_q75, r_q95 = np.percentile(r_paths,   [5, 25, 50, 75, 95], axis=0)
        p_q05, p_q25, p_med, p_q75, p_q95 = np.percentile(phi_paths, [5, 25, 50, 75, 95], axis=0)

        r_q05 = np.concatenate([[last_r],   r_q05])
        r_q25 = np.concatenate([[last_r],   r_q25])
        r_med = np.concatenate([[last_r],   r_med])
        r_q75 = np.concatenate([[last_r],   r_q75])
        r_q95 = np.concatenate([[last_r],   r_q95])

        p_q05 = np.concatenate([[last_phi], p_q05])
        p_q25 = np.concatenate([[last_phi], p_q25])
        p_med = np.concatenate([[last_phi], p_med])
        p_q75 = np.concatenate([[last_phi], p_q75])
        p_q95 = np.concatenate([[last_phi], p_q95])

        standalone = axes is None
        if standalone:
            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        axes[0].plot(data.index, data['r'],   color='gray', alpha=0.2, linewidth=0.6)
        axes[0].scatter(dates[pos_r],  r_vals[pos_r],  color=COLORS[1], s=6, alpha=0.7, label='c\u2083 > 0')
        axes[0].scatter(dates[~pos_r], r_vals[~pos_r], color=COLORS[5], s=6, alpha=0.7, label='c\u2083 \u2264 0')
        axes[0].fill_between(anchor_dates, r_q05, r_q95, color='red', alpha=0.10)
        axes[0].fill_between(anchor_dates, r_q25, r_q75, color='red', alpha=0.25)
        axes[0].plot(anchor_dates, r_med, color='red', linewidth=1.5, linestyle='--', zorder=3)
        axes[0].axvline(last_date, color='black', linewidth=1, linestyle=':', alpha=0.6)
        axes[0].axhline(0, color='black', linewidth=0.6, linestyle='--')
        axes[0].set_ylabel('r')
        axes[0].set_title(f'Log Return & \u03c6 — Past + {n_ahead}d Simulation ({n_paths} paths)')
        axes[0].legend(markerscale=3, loc='upper left')
        r_finite  = data['r'].dropna().values
        q1r, q99r = np.percentile(r_finite, [1, 99])
        axes[0].set_ylim(q1r - (q99r - q1r) * 0.15, q99r + (q99r - q1r) * 0.15)

        axes[1].plot(data.index, data['phi'], color='gray', alpha=0.2, linewidth=0.6)
        axes[1].scatter(dates[pos_phi],  phi_vals[pos_phi],  color=COLORS[1], s=6, alpha=0.7)
        axes[1].scatter(dates[~pos_phi], phi_vals[~pos_phi], color=COLORS[5], s=6, alpha=0.7)
        axes[1].fill_between(anchor_dates, p_q05, p_q95, color='red', alpha=0.10)
        axes[1].fill_between(anchor_dates, p_q25, p_q75, color='red', alpha=0.25)
        axes[1].plot(anchor_dates, p_med, color='red', linewidth=1.5, linestyle='--', zorder=3)
        axes[1].axvline(last_date, color='black', linewidth=1, linestyle=':', alpha=0.6)
        axes[1].axhline(0, color='black', linewidth=0.6, linestyle='--')
        axes[1].set_ylabel('\u03c6')
        phi_finite  = data['phi'].dropna().values
        q1p, q99p   = np.percentile(phi_finite, [1, 99])
        axes[1].set_ylim(q1p - (q99p - q1p) * 0.15, q99p + (q99p - q1p) * 0.15)
        self._format_date_axis(axes[1])

        if standalone:
            plt.tight_layout()
            return fig

    def plot_phase_space(self, ax=None):
        state    = self.model.state_df
        c2_cols  = [c for c in state.columns if c.startswith('c2_')]
        c3_cols  = [c for c in state.columns if c.startswith('c3_')]
        cpm_cols = [c for c in state.columns if c.startswith('cpm_')]

        c2  = state[c2_cols].sum(axis=1).values
        c3  = state[c3_cols].sum(axis=1).values
        cpm = state[cpm_cols].sum(axis=1).values

        r     = np.sqrt(c2 ** 2 + c3 ** 2 + cpm ** 2)
        n     = len(r)
        r_pct = (np.argsort(np.argsort(r)) + 1).astype(float) / n

        # spherical decomposition — inclination from cpm axis, azimuth in c2-c3 plane
        theta = np.arccos(np.clip(cpm / (r + 1e-12), -1.0, 1.0))
        phi   = np.arctan2(c3, c2)

        xs = r_pct * np.sin(theta) * np.cos(phi)
        ys = r_pct * np.sin(theta) * np.sin(phi)
        zs = r_pct * np.cos(theta)

        standalone = ax is None
        if standalone:
            fig = plt.figure(figsize=(6, 6))
            ax  = fig.add_subplot(111, projection='3d')

        ax.scatter(xs[:-1], ys[:-1], zs[:-1],
                   c=np.arange(n - 1), cmap='viridis', s=6, alpha=0.5)
        ax.scatter(xs[-1], ys[-1], zs[-1],
                   s=80, color=COLORS[0], zorder=5, label='current')

        ax.set_xlabel('C\u2082')
        ax.set_ylabel('C\u2083')
        ax.set_zlabel('C\u00b1')
        ax.set_title('Distributional Phase Space', pad=15)
        ax.legend(markerscale=2)

        if standalone:
            return fig

    def plot_signal(self, ax=None):
        r   = self.model.results
        tau = self.model.tau

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(14, 4))

        ax.fill_between(r['date'], tau, 1.0, alpha=0.08, color=COLORS[1])
        ax.fill_between(r['date'], 0.5, tau, alpha=0.08, color=COLORS[2])
        ax.fill_between(r['date'], 0.0, 0.5, alpha=0.08, color=COLORS[5])

        ax.plot(r['date'], r['mlp_proba'], linewidth=0.9, alpha=0.9)
        ax.axhline(tau, color='black', linewidth=0.8, linestyle='--', label=f'\u03c4 = {tau}')
        ax.axhline(0.5, color='black', linewidth=0.4, linestyle=':')
        ax.set_ylabel('P(up)')
        ax.set_title('MLP Confidence')
        ax.legend()
        self._format_date_axis(ax)

        # set ylim after all draws so zones don't anchor the axis to [0, 1]
        p   = r['mlp_proba']
        pad = (p.max() - p.min()) * 0.15
        ax.set_ylim(max(0.0, p.min() - pad), min(1.0, p.max() + pad))

        if standalone:
            plt.tight_layout()
            return fig

    def plot_all(self, n_ahead=20, n_paths=200, view_steps=30):
        LEFT    = 0.07
        RIGHT   = 0.02
        TOP     = 0.04
        BOT     = 0.05
        H_GAP   = 0.05
        V_GAP   = 0.03

        LEFT_W  = 0.58
        FULL_W  = 1.0 - LEFT - RIGHT
        RIGHT_W = FULL_W - LEFT_W - H_GAP
        RIGHT_X = LEFT + LEFT_W + H_GAP

        weights = [1.2, 1.2, 1.0]
        n_rows  = len(weights)
        avail   = 1.0 - TOP - BOT - (n_rows - 1) * V_GAP
        unit    = avail / sum(weights)
        row_h   = [w * unit for w in weights]

        row_y             = [0.0] * n_rows
        row_y[n_rows - 1] = BOT
        for i in range(n_rows - 2, -1, -1):
            row_y[i] = row_y[i + 1] + row_h[i + 1] + V_GAP

        fig = plt.figure(figsize=(18, 12))

        ax_r   = self._make_ax(fig, LEFT,    LEFT_W,  row_y, row_h, 0, V_GAP)
        ax_phi = self._make_ax(fig, LEFT,    LEFT_W,  row_y, row_h, 1, V_GAP)
        ax_ps  = self._make_ax(fig, RIGHT_X, RIGHT_W, row_y, row_h, 0, V_GAP,
                               rowspan=2, projection='3d')
        ax_sig = self._make_ax(fig, LEFT,    FULL_W,  row_y, row_h, 2, V_GAP)

        ax_phi.sharex(ax_r)

        self.plot_features(axes=[ax_r, ax_phi], n_ahead=n_ahead, n_paths=n_paths)
        self.plot_phase_space(ax=ax_ps)
        self.plot_signal(ax=ax_sig)

        idx        = self.model.data.index
        last_date  = idx[-1]
        view_start = idx[max(0, len(idx) - view_steps)]
        future_end = self._future_trading_dates(last_date, n_ahead)[-1]

        ax_r.set_xlim(view_start, future_end + pd.Timedelta(days=5))
        ax_sig.set_xlim(view_start, last_date + pd.Timedelta(days=5))

        return fig