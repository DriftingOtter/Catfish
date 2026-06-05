import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from catfish.Viz import PlotTheme


class Plotter:

    def __init__(self, model):
        self.model = model

    def _state_emission_means(self):
        hmm = self.model.model
        if isinstance(hmm, GaussianHMM):
            return hmm.means_
        weights = hmm.weights_
        means   = hmm.means_
        return np.einsum('sm,smf->sf', weights, means)

    def _state_emission_variances(self):
        hmm = self.model.model
        if isinstance(hmm, GaussianHMM):
            return np.array([np.diag(hmm.covars_[s]) for s in range(hmm.n_components)])

        weights     = hmm.weights_
        means       = hmm.means_
        state_means = np.einsum('sm,smf->sf', weights, means)
        n_states    = hmm.n_components
        variances   = np.zeros_like(state_means)

        for s in range(n_states):
            for m in range(hmm.n_mix):
                w     = weights[s, m]
                mu_m  = means[s, m]
                sig_m = np.diag(hmm.covars_[s, m])
                variances[s] += w * (sig_m + (mu_m - state_means[s]) ** 2)

        return variances

    def _forward_project(self, n_ahead):
        # state probability evolution via transition matrix
        transmat      = self.model.get_transition_matrix()
        current_probs = self.model.get_current_regime_distribution()
        n_states      = self.model.model.n_components

        state_probs = np.zeros((n_ahead, n_states))
        probs       = current_probs.copy()

        for t in range(n_ahead):
            probs = probs @ transmat
            state_probs[t] = probs

        return state_probs

    def _simulate_paths(self, n_ahead, n_paths=200, seed=0):
        rng      = np.random.default_rng(seed)
        hmm      = self.model.model
        n_states = hmm.n_components
        transmat = self.model.get_transition_matrix()
        current  = self.model.get_current_regime_distribution()

        emissions = np.zeros((n_paths, n_ahead, 2))

        for p in range(n_paths):
            state = rng.choice(n_states, p=current)
            for t in range(n_ahead):
                state = rng.choice(n_states, p=transmat[state])

                if isinstance(hmm, GaussianHMM):
                    emission = rng.multivariate_normal(hmm.means_[state], hmm.covars_[state])
                else:
                    mix      = rng.choice(hmm.n_mix, p=hmm.weights_[state])
                    emission = rng.multivariate_normal(hmm.means_[state, mix], hmm.covars_[state, mix])

                emissions[p, t] = emission

        # inverse-transform back to original space
        flat      = emissions.reshape(-1, 2)
        flat_orig = self.model.scaler.inverse_transform(flat)
        return flat_orig.reshape(n_paths, n_ahead, 2)

    def plot_price_with_regimes(self, ax=None, title="Close Price by Regime"):
        n_states = self.model.model.n_components
        data     = self.model.training_data.copy()
        data["state"] = self.model.get_state_path()

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(14, 5))

        PlotTheme.style_ax(ax)

        ax.plot(data.index, data["Close"], color=PlotTheme.TRACE, alpha=0.3, linewidth=0.8, zorder=1)
        for state in range(n_states):
            mask = data["state"] == state
            ax.scatter(data.index[mask], data["Close"][mask],
                       s=8, color=PlotTheme.STATE[state], label=f"State {state}", zorder=2)

        ax.set_ylabel("Close Price")
        ax.set_title(title)
        PlotTheme.legend(ax, loc='upper left', markerscale=2)
        PlotTheme.format_date_axis(ax)

        if standalone:
            plt.tight_layout()
            return fig

    def plot_regime_timeseries(self, axes=None, title="Log Return & Volume Proxy by Regime"):
        n_states = self.model.model.n_components
        data     = self.model.training_data.copy()
        data["state"] = self.model.get_state_path()

        standalone = axes is None
        if standalone:
            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        for ax in axes:
            PlotTheme.style_ax(ax)

        for state in range(n_states):
            mask = data["state"] == state
            axes[0].scatter(data.index[mask], data["r"][mask],   s=8, color=PlotTheme.STATE[state],
                            label=f"State {state}")
            axes[1].scatter(data.index[mask], data["phi"][mask], s=8, color=PlotTheme.STATE[state])

        axes[0].plot(data.index, data["r"],   color=PlotTheme.TRACE, alpha=0.2, linewidth=0.6)
        axes[1].plot(data.index, data["phi"], color=PlotTheme.TRACE, alpha=0.2, linewidth=0.6)

        axes[0].set_ylabel("Log Return (r)")
        axes[0].axhline(0, color=PlotTheme.ZERO, linewidth=0.6, linestyle='--')
        PlotTheme.legend(axes[0], loc='upper left', markerscale=2)
        axes[0].set_title(title)
        axes[1].set_ylabel("Signed Vol Proxy (φ)")
        axes[1].axhline(0, color=PlotTheme.ZERO, linewidth=0.6, linestyle='--')
        PlotTheme.format_date_axis(axes[1])
        r_finite  = data["r"].dropna().values
        q1r, q99r = np.percentile(r_finite, [1, 99])
        axes[0].set_ylim(q1r - (q99r - q1r) * 0.15, q99r + (q99r - q1r) * 0.15)
        phi_finite  = data["phi"].dropna().values
        q1p, q99p   = np.percentile(phi_finite, [1, 99])
        axes[1].set_ylim(q1p - (q99p - q1p) * 0.15, q99p + (q99p - q1p) * 0.15)

        if standalone:
            plt.tight_layout()
            return fig

    def plot_state_probabilities(self, ax=None, title="Posterior State Probabilities"):
        n_states = self.model.model.n_components
        probs    = self.model.model.predict_proba(self.model.feature_vector)
        dates    = self.model.training_data.index

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(14, 5))

        PlotTheme.style_ax(ax)

        ax.stackplot(dates,
                     [probs[:, i] for i in range(n_states)],
                     labels=[f"State {i}" for i in range(n_states)],
                     colors=PlotTheme.STATE[:n_states],
                     alpha=0.8)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        PlotTheme.legend(ax, loc='upper left')
        PlotTheme.format_date_axis(ax)

        if standalone:
            plt.tight_layout()
            return fig

    def plot_transition_matrix(self, ax=None, title="State Transition Matrix"):
        transmat = self.model.get_transition_matrix()
        n_states = self.model.model.n_components

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 6))

        PlotTheme.style_ax(ax)

        im = ax.imshow(transmat, cmap=PlotTheme.heatmap_cmap(), vmin=0, vmax=1)
        cb = plt.colorbar(im, ax=ax, label="Probability")
        PlotTheme.style_colorbar(cb, label="Probability")
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels([f"State {i}" for i in range(n_states)])
        ax.set_yticklabels([f"State {i}" for i in range(n_states)])
        ax.set_xlabel("To")
        ax.set_ylabel("From")
        ax.set_title(title)

        for i in range(n_states):
            for j in range(n_states):
                ax.text(j, i, f"{transmat[i, j]:.2f}",
                        ha='center', va='center',
                        color=PlotTheme.LEGEND if transmat[i, j] > 0.5 else PlotTheme.TICK,
                        fontsize=9)

        if standalone:
            plt.tight_layout()
            return fig

    def plot_state_statistics(self, axes=None, title="State Statistics"):
        n_states = self.model.model.n_components
        summary  = self.model.get_state_summary()
        states   = summary.index

        r_mean   = summary[("r",   "mean")]
        r_std    = summary[("r",   "std")]
        phi_mean = summary[("phi", "mean")]
        phi_std  = summary[("phi", "std")]
        counts   = summary[("r",   "count")]

        standalone = axes is None
        if standalone:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax in axes:
            PlotTheme.style_ax(ax)

        axes[0].bar(states, r_mean, yerr=r_std, capsize=5,
                    color=PlotTheme.STATE[:n_states], alpha=0.8)
        axes[0].axhline(0, color=PlotTheme.ZERO, linewidth=0.8, linestyle='--')
        axes[0].set_xticks(states)
        axes[0].set_xticklabels([f"State {s}" for s in states])
        axes[0].set_title("Mean Log Return (r)")
        axes[0].set_ylabel("r")

        axes[1].bar(states, phi_mean, yerr=phi_std, capsize=5,
                    color=PlotTheme.STATE[:n_states], alpha=0.8)
        axes[1].axhline(0, color=PlotTheme.ZERO, linewidth=0.8, linestyle='--')
        axes[1].set_xticks(states)
        axes[1].set_xticklabels([f"State {s}" for s in states])
        axes[1].set_title("Mean Signed Vol Proxy (φ)")
        axes[1].set_ylabel("φ")

        axes[2].bar(states, counts, color=PlotTheme.STATE[:n_states], alpha=0.8)
        axes[2].set_xticks(states)
        axes[2].set_xticklabels([f"State {s}" for s in states])
        axes[2].set_title("Days in State")
        axes[2].set_ylabel("Count")

        if standalone:
            PlotTheme.suptitle(title)
            plt.tight_layout()
            return fig

    def plot_forecast(self, n_ahead=20, n_paths=200, axes=None, title=""):
        n_states = self.model.model.n_components
        data     = self.model.training_data.copy()
        data["state"] = self.model.get_state_path()

        last_date    = data.index[-1]
        future_dates = PlotTheme.future_trading_dates(last_date, n_ahead)

        paths   = self._simulate_paths(n_ahead, n_paths=n_paths)
        r_paths = paths[:, :, 0]
        p_paths = paths[:, :, 1]

        standalone = axes is None
        if standalone:
            fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        for ax in axes:
            PlotTheme.style_ax(ax)

        for state in range(n_states):
            mask = data["state"] == state
            axes[0].scatter(data.index[mask], data["r"][mask],   s=8, color=PlotTheme.STATE[state],
                             label=f"State {state}", zorder=2)
            axes[1].scatter(data.index[mask], data["phi"][mask], s=8, color=PlotTheme.STATE[state],
                            zorder=2)

        axes[0].plot(data.index, data["r"],   color=PlotTheme.TRACE, alpha=0.2, linewidth=0.6)
        axes[1].plot(data.index, data["phi"], color=PlotTheme.TRACE, alpha=0.2, linewidth=0.6)

        # anchor forecast bands to last historical point for visual continuity
        anchor_dates = pd.DatetimeIndex([last_date]).append(future_dates)
        last_r       = float(data["r"].iloc[-1])
        last_phi     = float(data["phi"].iloc[-1])

        for ax, path_data, last_val in [
            (axes[0], r_paths, last_r),
            (axes[1], p_paths, last_phi),
        ]:
            q05, q25, med, q75, q95 = np.percentile(path_data, [5, 25, 50, 75, 95], axis=0)
            q05 = np.concatenate([[last_val], q05])
            q25 = np.concatenate([[last_val], q25])
            med = np.concatenate([[last_val], med])
            q75 = np.concatenate([[last_val], q75])
            q95 = np.concatenate([[last_val], q95])
            ax.fill_between(anchor_dates, q05, q95, color=PlotTheme.BAND, alpha=0.10)
            ax.fill_between(anchor_dates, q25, q75, color=PlotTheme.BAND, alpha=0.25)
            ax.plot(anchor_dates, med, color=PlotTheme.BAND, linewidth=1.5, linestyle='--', zorder=3)
            ax.axvline(last_date, color=PlotTheme.MARKER, linewidth=1, linestyle=':', alpha=0.6)
            ax.axhline(0, color=PlotTheme.ZERO, linewidth=0.6, linestyle='--')

        axes[0].set_ylabel("Log Return (r)")
        axes[0].set_title(title or f"r & φ — Past + {n_ahead}d Forecast ({n_paths} paths)")
        PlotTheme.legend(axes[0], loc='upper left', markerscale=2)
        axes[1].set_ylabel("Signed Vol Proxy (φ)")
        PlotTheme.format_date_axis(axes[1])
        r_finite  = data["r"].dropna().values
        q1r, q99r = np.percentile(r_finite, [1, 99])
        axes[0].set_ylim(q1r - (q99r - q1r) * 0.15, q99r + (q99r - q1r) * 0.15)
        phi_finite  = data["phi"].dropna().values
        q1p, q99p   = np.percentile(phi_finite, [1, 99])
        axes[1].set_ylim(q1p - (q99p - q1p) * 0.15, q99p + (q99p - q1p) * 0.15)

        if standalone:
            plt.tight_layout()
            return fig

    def plot_forecast_state_probs(self, n_ahead=20, ax=None, title=""):
        n_states   = self.model.model.n_components
        hist_probs = self.model.model.predict_proba(self.model.feature_vector)
        hist_dates = self.model.training_data.index
        last_date  = hist_dates[-1]

        future_dates = PlotTheme.future_trading_dates(last_date, n_ahead)
        future_probs = self._forward_project(n_ahead)

        all_dates = list(hist_dates) + list(future_dates)
        all_probs = np.vstack([hist_probs, future_probs])

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(16, 5))

        PlotTheme.style_ax(ax)

        ax.stackplot(all_dates,
                     [all_probs[:, i] for i in range(n_states)],
                     labels=[f"State {i}" for i in range(n_states)],
                     colors=PlotTheme.STATE[:n_states],
                     alpha=0.8)
        ax.axvline(last_date, color=PlotTheme.MARKER, linewidth=1.2, linestyle=':', alpha=0.8)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title(title or f"State Probabilities — Historical + {n_ahead}d Forecast")
        PlotTheme.legend(ax, loc='upper left')
        PlotTheme.format_date_axis(ax)

        if standalone:
            plt.tight_layout()
            return fig

    def plot_all(self, prefix="", n_ahead=20, n_paths=200, view_steps=30):
        LEFT   = 0.07
        RIGHT  = 0.02
        TOP    = 0.04
        BOT    = 0.05
        H_GAP  = 0.05
        V_GAP  = 0.03
        S_GAP  = 0.02

        LEFT_W  = 0.62
        FULL_W  = 1.0 - LEFT - RIGHT
        RIGHT_W = FULL_W - LEFT_W - H_GAP
        RIGHT_X = LEFT + LEFT_W + H_GAP
        STAT_W  = (FULL_W - 2 * S_GAP) / 3

        weights = [1.2, 1.2, 1.0, 1.2]
        n_rows  = len(weights)
        avail   = 1.0 - TOP - BOT - (n_rows - 1) * V_GAP
        unit    = avail / sum(weights)
        row_h   = [w * unit for w in weights]

        row_y             = [0.0] * n_rows
        row_y[n_rows - 1] = BOT
        for i in range(n_rows - 2, -1, -1):
            row_y[i] = row_y[i + 1] + row_h[i + 1] + V_GAP

        fig = plt.figure(figsize=(18, 14))
        PlotTheme.style_figure(fig)

        ax_fr   = PlotTheme.make_ax(fig, LEFT,                         LEFT_W,  row_y, row_h, 0, V_GAP)
        ax_fphi = PlotTheme.make_ax(fig, LEFT,                         LEFT_W,  row_y, row_h, 1, V_GAP)
        ax_tm   = PlotTheme.make_ax(fig, RIGHT_X,                      RIGHT_W, row_y, row_h, 0, V_GAP, rowspan=2)
        ax_fsp  = PlotTheme.make_ax(fig, LEFT,                         FULL_W,  row_y, row_h, 2, V_GAP)
        ax_sr   = PlotTheme.make_ax(fig, LEFT,                         STAT_W,  row_y, row_h, 3, V_GAP)
        ax_sphi = PlotTheme.make_ax(fig, LEFT + (STAT_W + S_GAP),      STAT_W,  row_y, row_h, 3, V_GAP)
        ax_sc   = PlotTheme.make_ax(fig, LEFT + 2 * (STAT_W + S_GAP), STAT_W,  row_y, row_h, 3, V_GAP)

        ax_fphi.sharex(ax_fr)

        self.plot_forecast(            axes=[ax_fr, ax_fphi], n_ahead=n_ahead, n_paths=n_paths)
        self.plot_transition_matrix(   ax=ax_tm)
        self.plot_forecast_state_probs(ax=ax_fsp, n_ahead=n_ahead)
        self.plot_state_statistics(    axes=[ax_sr, ax_sphi, ax_sc])

        idx        = self.model.training_data.index
        last_date  = idx[-1]
        view_start = idx[max(0, len(idx) - view_steps)]
        future_end = PlotTheme.future_trading_dates(last_date, n_ahead)[-1]

        ax_fr.set_xlim(view_start, future_end + pd.Timedelta(days=5))
        ax_fsp.set_xlim(view_start, future_end + pd.Timedelta(days=5))

        return fig
