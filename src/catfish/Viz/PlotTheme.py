from typing import final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BG:   final = '#080818'
SURF: final = '#0b0b22'
GRID: final = '#1a1a3a'

TICK:    final = '#8899bb'
LABEL:   final = '#9999bb'
TITLE:   final = '#bbbbdd'
LEGEND:  final = '#ccccee'
TWIN:    final = '#8888aa'
CB_TICK: final = '#777799'

HIST:      final = '#445599'
FORECAST:  final = '#a78bfa'
UNCERTAIN: final = '#7c3aed'
REF:       final = '#223388'
MARKER:    final = '#f87171'
WARNING:   final = '#3d0000'
ZERO:      final = '#223388'

EMERALD: final = '#34d399'
SKY:     final = '#38bdf8'
ORANGE:  final = '#fb923c'
TRACE:   final = '#445599'
BAND:    final = '#f87171'

STATE: final = ['#38bdf8', '#34d399', '#fb923c', '#f472b6', '#a78bfa', '#f87171']

DENSITY_STOPS: final = ['#04041a', '#1a0553', '#5b1acb', '#a78bfa', '#f0ecff']
HEATMAP_STOPS: final = ['#0b0b22', '#1a1a3a', '#3355aa', '#38bdf8', '#a0d8ef']


class PlotTheme:

    BG   = BG
    SURF = SURF
    GRID = GRID

    TICK    = TICK
    LABEL   = LABEL
    TITLE   = TITLE
    LEGEND  = LEGEND
    TWIN    = TWIN
    CB_TICK = CB_TICK

    HIST      = HIST
    FORECAST  = FORECAST
    UNCERTAIN = UNCERTAIN
    REF       = REF
    MARKER    = MARKER
    WARNING   = WARNING
    ZERO      = ZERO

    EMERALD = EMERALD
    SKY     = SKY
    ORANGE  = ORANGE
    TRACE   = TRACE
    BAND    = BAND

    STATE = STATE

    @staticmethod
    def style_ax(ax):
        ax.set_facecolor(SURF)
        ax.tick_params(colors=TICK, labelsize=8.5)
        ax.xaxis.label.set_color(LABEL)
        ax.yaxis.label.set_color(LABEL)
        ax.title.set_color(TITLE)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)

    @staticmethod
    def style_twin_ax(ax):
        ax.tick_params(colors=TWIN, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

    @staticmethod
    def style_ax_3d(ax):
        ax.set_facecolor(SURF)
        ax.tick_params(colors=TICK, labelsize=8)
        ax.xaxis.label.set_color(LABEL)
        ax.yaxis.label.set_color(LABEL)
        ax.zaxis.label.set_color(LABEL)
        ax.xaxis.pane.set_facecolor(SURF)
        ax.yaxis.pane.set_facecolor(SURF)
        ax.zaxis.pane.set_facecolor(SURF)
        ax.xaxis.pane.set_edgecolor(GRID)
        ax.yaxis.pane.set_edgecolor(GRID)
        ax.zaxis.pane.set_edgecolor(GRID)

    @staticmethod
    def style_figure(fig):
        fig.set_facecolor(BG)

    @staticmethod
    def style_colorbar(cb, label=None, labelsize=7):
        cb.ax.tick_params(labelsize=labelsize, colors=CB_TICK)
        if label is not None:
            cb.set_label(label, fontsize=labelsize, color=CB_TICK)

    @staticmethod
    def legend(ax, *args, **kwargs):
        defaults = dict(frameon=False, fontsize=8.5, labelcolor=LEGEND)
        defaults.update(kwargs)
        return ax.legend(*args, **defaults)

    @staticmethod
    def suptitle(text, **kwargs):
        defaults = dict(color=LEGEND, fontsize=13, fontweight='bold', y=0.975)
        defaults.update(kwargs)
        plt.suptitle(text, **defaults)

    @staticmethod
    def density_cmap(name='catfish', n=256):
        return LinearSegmentedColormap.from_list(name, DENSITY_STOPS, N=n)

    @staticmethod
    def heatmap_cmap(name='catfish_heat', n=256):
        return LinearSegmentedColormap.from_list(name, HEATMAP_STOPS, N=n)

    @staticmethod
    def format_date_axis(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    @staticmethod
    def future_trading_dates(last_date, n_ahead):
        return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_ahead)

    @staticmethod
    def make_ax(fig, col_x, col_w, row_y, row_h, row, v_gap, rowspan=1, projection=None):
        last = row + rowspan - 1
        b    = row_y[last]
        h    = sum(row_h[row:last + 1]) + (rowspan - 1) * v_gap
        return fig.add_axes([col_x, b, col_w, h], projection=projection)
