import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from catfish.Viz import PlotTheme


class Plotter:

    def __init__(self, model):
        self.model = model

    def plot_displacement(self, ax=None):
        if self.model.x_hist is None:
            raise RuntimeError("Features not calculated.")
        if self.model.forecast_result is None:
            raise RuntimeError("Forecast not computed.")

        fc     = self.model.forecast_result
        x_hist = self.model.x_hist
        xE     = fc['xE']
        sig    = fc['sigma']

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(14, 5))

        PlotTheme.style_ax(ax)

        t_h = np.arange(len(x_hist))
        t_f = np.arange(len(x_hist), len(x_hist) + len(xE))

        ax.plot(t_h, x_hist, color=PlotTheme.HIST, lw=0.8, alpha=0.75,
                label='x = P − VWAP  (historical)')
        ax.plot(t_f, xE, color=PlotTheme.FORECAST, lw=2.0, label='⟨x⟩  forecast')
        ax.fill_between(t_f, xE - sig, xE + sig,
                        color=PlotTheme.UNCERTAIN, alpha=0.28, label='±1σ')
        ax.fill_between(t_f, xE - 2 * sig, xE + 2 * sig,
                        color=PlotTheme.UNCERTAIN, alpha=0.10, label='±2σ')
        ax.axhline(0, color=PlotTheme.ZERO, lw=0.9, ls='--', alpha=0.5, label='VWAP')
        ax.axvline(len(x_hist), color=PlotTheme.MARKER, lw=1.3, ls=':', alpha=0.9,
                   label='Forecast origin')

        if self.model.gex_eff < 0:
            ax.axvspan(len(x_hist), len(x_hist) + len(xE),
                       color=PlotTheme.WARNING, alpha=0.15)

        regime = 'mean-reverting  ∪' if self.model.gex_eff >= 0 else 'trending  ∩'
        ax.set_title(
            f'{self.model.ticker}  ·  BDD Quantum Forecast  ·  '
            f'GEX_eff = {self.model.gex_eff:+.2f}  ({regime})   '
            f'α = {self.model.alpha:.2f}   ħ_eff = {self.model.h_eff:.3f}',
            fontsize=10.0, pad=8)
        ax.set_ylabel('Price displacement  x  ($)')
        ax.set_xlabel('Time step')
        PlotTheme.legend(ax, loc='upper left', ncol=6)

        if standalone:
            plt.tight_layout()
            return fig

    def plot_density(self, ax=None):
        if self.model.forecast_result is None:
            raise RuntimeError("Forecast not computed.")

        fc    = self.model.forecast_result
        xE    = fc['xE']
        snaps = fc['snaps']

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        PlotTheme.style_ax(ax)

        sg = max(1, self.model.N // 200)
        gd = self.model.grid[::sg]
        sd = snaps[:, ::sg]
        ts = np.linspace(0, len(xE) - 1, len(snaps))
        im = ax.pcolormesh(ts, gd, sd.T, cmap=PlotTheme.density_cmap(), shading='auto')
        ax.plot(np.arange(len(xE)), xE, color=PlotTheme.ORANGE, lw=1.6, label='⟨x⟩')
        ax.set_xlabel('Forecast step')
        ax.set_ylabel('x = P − VWAP  ($)')
        ax.set_title('|ψ(x, t)|²  probability density', fontsize=9.5)
        PlotTheme.legend(ax, fontsize=8, labelcolor=PlotTheme.ORANGE, loc='upper right')
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        PlotTheme.style_colorbar(cb, label='ρ = |ψ|²')

        if standalone:
            plt.tight_layout()
            return fig

    def plot_mass(self, ax=None):
        if self.model.forecast_result is None:
            raise RuntimeError("Forecast not computed.")

        fc = self.model.forecast_result

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        PlotTheme.style_ax(ax)

        ax.plot(self.model.grid, self.model.mass, color=PlotTheme.EMERALD, lw=2.0, label='m(x)')
        ax.fill_between(self.model.grid, 1.0, self.model.mass,
                        color=PlotTheme.EMERALD, alpha=0.13)
        ax.axvline(0, color=PlotTheme.REF, lw=0.8, ls='--', alpha=0.4)
        axb = ax.twinx()
        axb.plot(self.model.grid, fc['rho_final'], color=PlotTheme.FORECAST,
                 lw=1.5, ls='--', alpha=0.85, label='|ψ|²  (final)')
        axb.set_ylabel('|ψ|²', fontsize=8.5)
        PlotTheme.style_twin_ax(axb)
        ax.set_xlabel('x = P − VWAP  ($)')
        ax.set_ylabel('Market mass  m(x)', color=PlotTheme.EMERALD, fontsize=8.5)
        ax.tick_params(axis='y', colors=PlotTheme.EMERALD)
        ax.set_title('Latent mass  m(x) = 1 + α(x/ħ)²', fontsize=9.5)
        ax.set_xlim(-4.0 * self.model.h_eff, 4.0 * self.model.h_eff)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        PlotTheme.legend(ax, h1 + h2, l1 + l2, fontsize=8, loc='upper center')

        if standalone:
            plt.tight_layout()
            return fig

    def plot_observables(self, ax=None):
        if self.model.forecast_result is None:
            raise RuntimeError("Forecast not computed.")

        fc  = self.model.forecast_result
        sig = fc['sigma']
        pE  = fc['pE']

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        PlotTheme.style_ax(ax)

        ts = np.arange(len(sig))
        ax.plot(ts, sig, color=PlotTheme.EMERALD, lw=1.8, label='σ_x  (price spread)')
        axb = ax.twinx()
        axb.plot(ts, pE, color=PlotTheme.SKY, lw=1.4, ls='--',
                 alpha=0.85, label='⟨p⟩  (momentum)')
        axb.axhline(0, color=PlotTheme.REF, lw=0.7, ls=':', alpha=0.5)
        axb.set_ylabel('⟨p⟩', fontsize=8.5)
        PlotTheme.style_twin_ax(axb)
        ax.set_xlabel('Forecast step')
        ax.set_ylabel('σ_x  ($)', color=PlotTheme.EMERALD, fontsize=8.5)
        ax.tick_params(axis='y', colors=PlotTheme.EMERALD)
        ax.set_title('Spread  &  momentum  evolution', fontsize=9.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        PlotTheme.legend(ax, h1 + h2, l1 + l2, fontsize=8, loc='upper left')

        if standalone:
            plt.tight_layout()
            return fig

    def plot_all(self, axes=None):
        if self.model.x_hist is None:
            raise RuntimeError("Features not calculated.")
        if self.model.forecast_result is None:
            raise RuntimeError("Forecast not computed.")

        standalone = axes is None
        if standalone:
            fig = plt.figure(figsize=(14, 8.5))
            PlotTheme.style_figure(fig)
            gs  = gridspec.GridSpec(
                2, 3, figure=fig,
                height_ratios=[1.55, 1.0],
                hspace=0.44, wspace=0.36,
                left=0.07, right=0.97,
                top=0.91, bottom=0.09,
            )
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[1, 2])
        else:
            ax1, ax2, ax3, ax4 = axes
            fig = ax1.figure

        self.plot_displacement(ax=ax1)
        self.plot_density(ax=ax2)
        self.plot_mass(ax=ax3)
        self.plot_observables(ax=ax4)

        if standalone:
            PlotTheme.suptitle(
                f'BenDaniel–Duke Market Model  ·  {self.model.ticker}',
            )
            plt.tight_layout()
            return fig
