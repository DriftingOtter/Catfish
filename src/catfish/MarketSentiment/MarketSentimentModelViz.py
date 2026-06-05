import enum
from functools import partial
from typing import final

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from catfish.Viz import PlotTheme

SENTIMENT_ORDER: final = ["positive", "neutral", "negative"]
SENTIMENT_COLORS: final = {
    "positive": PlotTheme.EMERALD,
    "neutral":  PlotTheme.SKY,
    "negative": PlotTheme.MARKER,
}
CHECK_SIZE: final = 0.028


class SortMode(enum.Enum):
    Type     = 0
    Date     = 1
    DateType = 2


class Plotter:

    def __init__(self, model):
        self.model = model

        self._fig          = None
        self._ax_bar       = None
        self._ax_tbl       = None
        self._row_bboxes   = []
        self._selected     = 0
        self._sort_mode    = SortMode.DateType
        self._cid          = None

    def plot_sentiment(self, sentiment, filing=None, ax=None):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 4))
            PlotTheme.style_figure(fig)

        PlotTheme.style_ax(ax)

        score_map = {label.lower(): score for label, score in sentiment}
        labels    = [label.capitalize() for label in SENTIMENT_ORDER]
        scores    = [score_map.get(label, 0.0) for label in SENTIMENT_ORDER]
        colors    = [SENTIMENT_COLORS[label] for label in SENTIMENT_ORDER]

        y = range(len(labels))
        ax.barh(y, scores, color=colors, height=0.55, alpha=0.88)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score (%)")
        ax.invert_yaxis()

        title = "FinBERT Sentiment"
        if filing is not None:
            title = (
                f"{self.model.symbol}  ·  {filing['doc_type']}  "
                f"{filing['year']}-{filing['month']:02d}-{filing['day']:02d}"
            )
        ax.set_title(title)

        for i, score in enumerate(scores):
            ax.text(score + 1.2, i, f"{score:.1f}%", va='center',
                    color=PlotTheme.LEGEND, fontsize=9)

        if standalone:
            plt.tight_layout()
            return fig

    def _sorted_filings(self):
        filings = list(self.model.filings)

        if self._sort_mode == SortMode.Date:
            filings.sort(key=lambda f: f["date"].toordinal(), reverse=True)
        elif self._sort_mode == SortMode.Type:
            filings.sort(key=lambda f: (f["doc_type"], -f["date"].toordinal()))
        else:
            filings.sort(key=lambda f: (-f["date"].toordinal(), f["doc_type"]))

        return filings

    def _draw_bar(self, sentiment, filing):
        self._ax_bar.clear()
        self.plot_sentiment(sentiment, filing=filing, ax=self._ax_bar)

    def _is_analysed(self, filing):
        return str(filing["path"]) in self.model.sentiments

    @staticmethod
    def _draw_checkbox(ax, x, y, checked, accent):
        size = CHECK_SIZE
        ax.add_patch(plt.Rectangle(
            (x, y - size), size, size,
            transform=ax.transAxes,
            facecolor=PlotTheme.FORECAST if checked else PlotTheme.SURF,
            edgecolor=accent if checked else PlotTheme.GRID,
            linewidth=1.0,
        ))
        if checked:
            ax.plot(
                [x + size * 0.18, x + size * 0.38, x + size * 0.82],
                [y - size * 0.52, y - size * 0.78, y - size * 0.22],
                color=accent, linewidth=1.4, solid_capstyle='round',
                transform=ax.transAxes, clip_on=False,
            )

    def _draw_table(self, filings, selected):
        self._ax_tbl.clear()
        self._ax_tbl.axis("off")
        self._ax_tbl.set_xlim(0, 1)
        self._ax_tbl.set_ylim(0, 1)

        headers = ["", "Type", "Year", "Month", "Day"]
        n       = len(filings)
        row_h   = min(0.07, 0.88 / max(n, 1))
        top     = 0.96

        col_x = [0.02, 0.14, 0.36, 0.56, 0.76]
        for j, header in enumerate(headers):
            if j == 0:
                self._draw_checkbox(
                    self._ax_tbl, col_x[j], top - row_h * 0.05, False, PlotTheme.TICK,
                )
                continue
            self._ax_tbl.text(
                col_x[j], top, header,
                color=PlotTheme.TITLE, fontsize=9, fontweight='bold',
                transform=self._ax_tbl.transAxes, va='top',
            )

        self._row_bboxes = []
        for i, filing in enumerate(filings):
            y_top = top - (i + 1) * row_h
            y_bot = y_top - row_h * 0.82
            self._row_bboxes.append((0.01, y_bot, 0.99, y_top))

            if i == selected:
                self._ax_tbl.add_patch(plt.Rectangle(
                    (0.01, y_bot), 0.98, y_top - y_bot,
                    transform=self._ax_tbl.transAxes,
                    facecolor=PlotTheme.FORECAST, alpha=0.18,
                    edgecolor=PlotTheme.FORECAST, linewidth=0.8,
                ))

            row_color = PlotTheme.LEGEND if i == selected else PlotTheme.TICK
            analysed  = self._is_analysed(filing)
            accent    = PlotTheme.EMERALD if analysed else PlotTheme.TICK

            self._draw_checkbox(
                self._ax_tbl, col_x[0], y_top - row_h * 0.12, analysed, accent,
            )

            values = [
                filing["doc_type"],
                str(filing["year"]),
                f"{filing['month']:02d}",
                f"{filing['day']:02d}",
            ]
            for j, value in enumerate(values):
                self._ax_tbl.text(
                    col_x[j + 1], y_top - row_h * 0.12, value,
                    color=row_color, fontsize=8.5,
                    transform=self._ax_tbl.transAxes, va='top',
                )

        self._ax_tbl.text(
            0.02, 0.02, "Click a row to analyse",
            color=PlotTheme.TICK, fontsize=7.5,
            transform=self._ax_tbl.transAxes, va='bottom',
        )

    def _row_from_click(self, event):
        if event.inaxes != self._ax_tbl:
            return None

        x, y = self._ax_tbl.transAxes.inverted().transform((event.x, event.y))
        for i, (x0, y0, x1, y1) in enumerate(self._row_bboxes):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return None

    def _on_click(self, event):
        row = self._row_from_click(event)
        if row is None:
            return

        filings = self._sorted_filings()
        filing  = filings[row]
        sentiment = self.model.analyse_filing(filing)

        self._selected = row
        self._draw_bar(sentiment, filing)
        self._draw_table(filings, row)
        self._fig.canvas.draw_idle()

    def _set_sort(self, mode, _event=None):
        self._sort_mode = mode
        filings = self._sorted_filings()

        active_path = None
        if self.model.active is not None:
            active_path = str(self.model.active["path"])

        self._selected = 0
        if active_path is not None:
            for i, filing in enumerate(filings):
                if str(filing["path"]) == active_path:
                    self._selected = i
                    break

        sentiment = self.model.analyse_filing(filings[self._selected])
        self._draw_bar(sentiment, filings[self._selected])
        self._draw_table(filings, self._selected)
        self._highlight_sort_buttons()
        self._fig.canvas.draw_idle()

    @staticmethod
    def _style_sort_button(btn, active):
        color = PlotTheme.FORECAST if active else PlotTheme.SURF
        btn.color = color
        btn.hovercolor = PlotTheme.GRID
        btn.label.set_color(PlotTheme.LEGEND)
        btn.ax.set_facecolor(color)

    def _highlight_sort_buttons(self):
        self._style_sort_button(self._btn_type,      self._sort_mode == SortMode.Type)
        self._style_sort_button(self._btn_date,      self._sort_mode == SortMode.Date)
        self._style_sort_button(self._btn_date_type, self._sort_mode == SortMode.DateType)

    def plot_all(self, sort_mode=SortMode.DateType):
        if not self.model.filings:
            raise RuntimeError("Filings not loaded.")

        if not isinstance(sort_mode, SortMode):
            raise ValueError("sort_mode must be a SortMode enum value.")

        self._sort_mode = sort_mode
        filings = self._sorted_filings()

        sentiment = self.model.analyse_filing(filings[0])
        self._selected = 0

        self._fig = plt.figure(figsize=(14, 7))
        PlotTheme.style_figure(self._fig)

        self._ax_bar = self._fig.add_axes([0.07, 0.10, 0.48, 0.82])
        self._ax_tbl = self._fig.add_axes([0.58, 0.10, 0.35, 0.75])

        self._draw_bar(sentiment, filings[0])
        self._draw_table(filings, 0)

        PlotTheme.suptitle(f"SEC Filing Sentiment  ·  {self.model.symbol}")

        btn_y = 0.87
        btn_h = 0.04
        self._btn_type = Button(
            self._fig.add_axes([0.58, btn_y, 0.10, btn_h]),
            "Type", color=PlotTheme.SURF, hovercolor=PlotTheme.GRID,
        )
        self._btn_date = Button(
            self._fig.add_axes([0.69, btn_y, 0.10, btn_h]),
            "Date", color=PlotTheme.SURF, hovercolor=PlotTheme.GRID,
        )
        self._btn_date_type = Button(
            self._fig.add_axes([0.80, btn_y, 0.13, btn_h]),
            "Date + Type", color=PlotTheme.SURF, hovercolor=PlotTheme.GRID,
        )

        self._btn_type.on_clicked(partial(self._set_sort, SortMode.Type))
        self._btn_date.on_clicked(partial(self._set_sort, SortMode.Date))
        self._btn_date_type.on_clicked(partial(self._set_sort, SortMode.DateType))
        self._highlight_sort_buttons()

        self._cid = self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        return self._fig
