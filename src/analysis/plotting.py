"""Reusable plotting helpers for analysis notebooks.

Public API
----------
Utilities:       set_analysis_theme, apply_science_style, set_publication_style, set_science_publication_style, save_plot, DisplayNames
Side-by-side:    render_de_delta_pair, save_panels
Overview:        plot_ie_adherence_overview
Per-IE:          plot_ie_rubric_heatmap
Country delta:   plot_country_delta_heatmap
Sub-question:    plot_subquestion_adherence
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter

# Signed percent formatter for delta colorbars and axes — shows "+10%" / "−10%"
# so the zero-centred scale is immediately readable as positive/negative.
_SIGNED_PCT_FORMATTER = FuncFormatter(
    lambda v, _: "0%" if v == 0 else f"{v * 100:+.0f}%"
)

NOTEBOOK_PREFIX_PATTERN = re.compile(r"^\d+[_-]*")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
SCIENCE_STYLES_BASE: tuple[str, ...] = ("science", "ieee")
SCIENCE_GRID_STYLE: str = "grid"

# Shared exploratory theme — every analysis notebook calls set_analysis_theme()
# so all on-screen figures look identical.
EXPLORATORY_STYLE: str = "whitegrid"
EXPLORATORY_CONTEXT: str = "notebook"
EXPLORATORY_PALETTE: str = "colorblind"


@dataclass(frozen=True)
class DisplayNames:
    """Central registry for human-readable display names in analysis plots.

    Attributes:
        party: Mapping from raw party IDs to display labels.
        model: Mapping from raw model IDs to display labels.
        ie: Mapping from raw intervention IDs to display labels.
        rubric: Mapping from raw rubric IDs to display labels.
        subquestion: Mapping from raw sub-question IDs to display labels.
    """

    party: dict[str, str] = field(default_factory=dict)
    model: dict[str, str] = field(default_factory=dict)
    ie: dict[str, str] = field(default_factory=dict)
    rubric: dict[str, str] = field(default_factory=dict)
    subquestion: dict[str, str] = field(default_factory=dict)

    def get(self, category: str) -> dict[str, str]:
        """Return the naming map for one category.

        Args:
            category: One of ``party``, ``model``, ``ie``, ``rubric``, or
                ``subquestion``.

        Returns:
            The mapping dictionary for the selected category.

        Raises:
            ValueError: If category is unknown.
        """

        mapping_table: dict[str, dict[str, str]] = {
            "party": self.party,
            "model": self.model,
            "ie": self.ie,
            "rubric": self.rubric,
            "subquestion": self.subquestion,
        }
        if category not in mapping_table:
            raise ValueError(f"Unknown display-name category: {category}")
        return mapping_table[category]

    def apply(self, value: str, category: str) -> str:
        """Map raw ID to display name, falling back to raw value."""

        return self.get(category).get(value, value)


def set_analysis_theme(*, context: str = EXPLORATORY_CONTEXT) -> None:
    """Apply the shared exploratory seaborn theme used across all notebooks.

    Clean white grid, on-screen element scaling, and a colourblind-safe
    qualitative palette. Call once at the top of every analysis notebook so
    figures stay visually consistent. For publication figures use
    :func:`set_publication_style` instead.

    Resets rcParams to the matplotlib defaults first, so a prior
    :func:`set_publication_style` call in the same kernel can never leak its
    font / tick / colour settings into the exploratory figures — the two styles
    stay fully isolated regardless of call order.

    Args:
        context: Seaborn context controlling element scale
            (``paper`` < ``notebook`` < ``talk`` < ``poster``).
    """

    plt.rcdefaults()  # hard reset — isolate from any prior set_publication_style()
    sns.set_theme(
        style=EXPLORATORY_STYLE,
        context=context,
        palette=EXPLORATORY_PALETTE,
    )


def apply_science_style(*, with_grid: bool = False) -> None:
    """Apply SciencePlots + seaborn-friendly base style.

    Args:
        with_grid: Include SciencePlots ``grid`` style when True.
    """

    import scienceplots  # noqa: F401

    styles: list[str] = list(SCIENCE_STYLES_BASE)
    if with_grid:
        styles.append(SCIENCE_GRID_STYLE)
    plt.style.use(styles)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLICATION STYLE (final vector figures)
# ═══════════════════════════════════════════════════════════════════════════

# Figure widths (inches) for single-column and full-width layout.
COLUMN_WIDTH_IN: float = 3.031
TEXT_WIDTH_IN: float = 6.30
GOLDEN_RATIO: float = 1.618

# Fixed semantic colour mapping — reused across ALL final figures so a colour
# means the same thing everywhere. Standard, CVD-safe matplotlib maps (same as
# the exploratory figures): viridis for sequential adherence, RdBu_r for
# zero-centred deltas.
SEQUENTIAL_CMAP: str = "viridis"  # adherence / pass-rate (sequential)
DIVERGING_CMAP: str = "RdBu_r"    # deltas centred at zero (diverging)

# Fixed adherence colour scale: clip the low end so real variation in the
# near-ceiling 75–100% band is visible, while staying constant across figures.
ADHERENCE_VMIN: float = 0.60
ADHERENCE_VMAX: float = 1.0

# Border drawn on cells flagged for notable model/party spread.
FLAG_EDGE_COLOR: str = "#111111"

# Fill for inactive / not-scored cells (e.g. Faithfulness & Impartiality under
# the Absent condition). NaN cells render in this neutral grey via the heatmap
# mask, signalling "not measured" rather than "scored zero".
INACTIVE_GREY: str = "#dcdcdc"

# In-cell / on-bar annotation font size. Centralised so all final figures stay consistent.
ANNOT_FONTSIZE: int = 7

# Colourblind-safe qualitative palette (seaborn "colorblind"), reused as the
# figure colour cycle so a model keeps the same colour as in the exploratory
# figures.
COLORBLIND_HEX: list[str] = sns.color_palette("colorblind").as_hex()

# Self-contained publication rcParams — NO SciencePlots base and NO usetex
# (that base shipped ``text.usetex:True`` + ``font.family:serif``, which forced
# the LaTeX Computer Modern / "Times" look and leaked into other figures). We
# replicate the few good bits of "science" explicitly: thin spines, no top/right
# spine, short ticks, frameless legend. Font family is injected per preset by
# set_publication_style(); everything else is fixed and machine-independent.
PUBLICATION_RCPARAMS: dict[str, Any] = {
    # No LaTeX engine: portable + fast, and renders the chosen TTF/OTF directly.
    "text.usetex": False,
    # Body-matched sizes — figures are generated at final width, never rescaled.
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,    # axis descriptor labels (xlabel/ylabel)
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,  # per-tick values and category names
    "ytick.labelsize": 7.5,
    # Minimal chrome with SciencePlots' crispness — thin 0.5 pt spines/ticks and
    # thin lines — but WITHOUT its weird parts (kept out, see comment above):
    # no inward ticks, no top/right ticks, no minor ticks, no grid.
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#aaaaaa",   # lighter spines — less stark than default black
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.color": "#ebebeb",       # slightly lighter than before
    "grid.linewidth": 0.4,         # a touch thinner
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,         # no tick nubs — label is sufficient
    "ytick.major.size": 0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
    # Familiar colourblind cycle (same as the exploratory figures).
    "axes.prop_cycle": plt.cycler(color=COLORBLIND_HEX),
    # Vector export with embedded TrueType fonts; rasterized meshes stay crisp.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 150,   # snappy inline preview
    "savefig.dpi": 600,  # crisp rasterized heatmap meshes in the PDF
    # Default single-column canvas (golden-ratio height); override per figure.
    "figure.figsize": (COLUMN_WIDTH_IN, COLUMN_WIDTH_IN / GOLDEN_RATIO),
}

# Font presets for publication figures. Primary is Nimbus Sans (URW Helvetica clone)
# because, unlike macOS Helvetica .ttc files, it registers a real bold face.
# System faces stay as fallbacks. Chosen via set_publication_style(font=).
PUBLICATION_FONTS: dict[str, tuple[str, list[str]]] = {
    # Sans-serif (current default)
    "helvetica": (
        "sans-serif",
        ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    ),
    # Computer Modern Serif — the actual LaTeX body font (requires cm-unicode / CMU)
    "cmu": (
        "serif",
        ["CMU Serif", "Computer Modern", "DejaVu Serif"],
    ),
    # STIX Two — modern serif, widely available
    "stix": (
        "serif",
        ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    ),
    # Palatino — classic high-legibility serif used in many conference proceedings
    "palatino": (
        "serif",
        ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    ),
    # Charter — Bitstream Charter, excellent on screen and in print
    "charter": (
        "serif",
        ["Charter", "Bitstream Charter", "DejaVu Serif"],
    ),
    # PT Serif — open-source, designed for mixed-script academic use
    "pt-serif": (
        "serif",
        ["PT Serif", "DejaVu Serif"],
    ),
}
DEFAULT_PUBLICATION_FONT: str = "helvetica"


def set_publication_style(font: str = DEFAULT_PUBLICATION_FONT) -> None:
    """Apply the locked publication style for final vector figures.

    Fully self-contained: resets rcParams to the matplotlib defaults, applies
    :data:`PUBLICATION_RCPARAMS` (no SciencePlots, no usetex), then injects the
    chosen font preset. Renders identically on any machine and keeps the final
    style isolated from the exploratory one. Call once before building final
    figures; use :func:`set_analysis_theme` for on-screen exploratory work.

    Args:
        font: Font preset key from :data:`PUBLICATION_FONTS`. Defaults to ``"helvetica"``.

    Raises:
        ValueError: If ``font`` is not a known preset.

    Usage (notebook):
        set_publication_style()
        fig = plot_emi_leaderboard(lb, country="DE", names=NAMES,
                                   colors=model_palette(NAMES))
        save_final_plot(fig=fig, repo_root=REPO_ROOT,
                        notebook_file=NOTEBOOK_FILE, plot_name="leaderboard")
    """

    if font not in PUBLICATION_FONTS:
        raise ValueError(
            f"Unknown font preset {font!r}; choose from {sorted(PUBLICATION_FONTS)}."
        )

    family, font_stack = PUBLICATION_FONTS[font]
    plt.rcdefaults()  # hard reset so repeated calls / prior themes never accumulate
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    plt.rcParams["font.family"] = family
    plt.rcParams["font.sans-serif" if family == "sans-serif" else "font.serif"] = font_stack
    plt.rcParams["mathtext.fontset"] = "dejavusans" if family == "sans-serif" else "cm"


def set_science_publication_style(font: str = DEFAULT_PUBLICATION_FONT) -> None:
    """Playground variant: SciencePlots ``["science", "no-latex"]`` base + our overrides.

    Layers SciencePlots' micro-typography improvements (tighter tick formatting,
    slightly bolder spines, cleaner grid) underneath our locked PUBLICATION_RCPARAMS,
    so our colours, sizes, and font choices win. Useful for side-by-side comparisons
    in the notebook before deciding whether to adopt any SciencePlots defaults.

    Key differences vs. :func:`set_publication_style`:
    - SciencePlots' ``science`` style tightens tick padding and uses slightly
      bolder axes lines before our 0.5 pt override.
    - ``no-latex`` prevents ``text.usetex=True`` so our font choice is preserved.
    - Everything in PUBLICATION_RCPARAMS (colours, font sizes, grid, export DPI)
      still applies — SciencePlots only affects the underlying defaults that our
      overrides don't explicitly set.

    Args:
        font: Font preset key (same as :func:`set_publication_style`).

    Usage (notebook — playground):
        set_science_publication_style()
        fig = plot_model_ie_robustness(df[df.country == "DE"], target_width=TEXT_WIDTH_IN)
        # compare with set_publication_style() version side by side
    """
    import scienceplots  # noqa: F401

    if font not in PUBLICATION_FONTS:
        raise ValueError(
            f"Unknown font preset {font!r}; choose from {sorted(PUBLICATION_FONTS)}."
        )
    family, font_stack = PUBLICATION_FONTS[font]
    plt.rcdefaults()
    plt.style.use(["science", "no-latex"])      # micro-typography base
    plt.rcParams.update(PUBLICATION_RCPARAMS)   # our settings win on conflict
    plt.rcParams["font.family"] = family
    plt.rcParams["font.sans-serif" if family == "sans-serif" else "font.serif"] = font_stack
    plt.rcParams["mathtext.fontset"] = "dejavusans" if family == "sans-serif" else "cm"


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════


def _figure_dir(
    repo_root: str | Path,
    notebook_file: str | Path,
    *,
    stage: str,
    subfolder: str | None = None,
) -> Path:
    """Resolve (and create) a figure output directory for a notebook focus.

    Args:
        repo_root: Repository root path containing ``analysis/``.
        notebook_file: Notebook path or filename (sets the focus folder).
        stage: ``"exploratory"`` (PNG) or ``"final"`` (vector PDF).
        subfolder: Optional sub-directory appended inside the stage folder.

    Returns:
        The resolved directory path (created if missing).
    """

    focus = notebook_focus_slug(notebook_file)
    output_dir = Path(repo_root) / "analysis" / "figures" / focus / stage
    if subfolder is not None:
        output_dir = output_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plot(
    *,
    fig: Any,
    repo_root: str | Path,
    notebook_file: str | Path,
    plot_name: str,
    dpi: int = 300,
    subfolder: str | None = None,
) -> dict[str, Path]:
    """Save plot in standard exploratory location.

    The output path is:
    - Without subfolder: ``analysis/figures/<focus>/exploratory/<name>.png``
    - With subfolder:    ``analysis/figures/<focus>/exploratory/<subfolder>/<name>.png``

    Args:
        fig: Matplotlib Figure object.
        repo_root: Repository root path containing ``analysis/``.
        notebook_file: Notebook path or filename.
        plot_name: Human-readable plot name.
        dpi: PNG DPI for exploratory output.
        subfolder: Optional sub-directory appended inside ``exploratory/``.
            When ``None``, behavior is unchanged.

    Returns:
        Dict with output path under key ``exploratory_png``.

    Usage (any notebook):
        saved = save_plot(fig=fig, repo_root=REPO_ROOT, notebook_file=NOTEBOOK_FILE,
                          plot_name="my_plot")
        # With per-IE subfolder → exploratory/baseline/<name>.png (notebook 02):
        saved = save_plot(fig=fig, repo_root=REPO_ROOT, notebook_file=NOTEBOOK_FILE,
                          plot_name="rubric_heatmap_baseline", subfolder="baseline")
    """

    exploratory_dir = _figure_dir(
        repo_root, notebook_file, stage="exploratory", subfolder=subfolder
    )
    exploratory_png = exploratory_dir / f"{plot_name_slug(plot_name)}.png"

    fig.savefig(exploratory_png, dpi=dpi, bbox_inches="tight")

    return {
        "exploratory_png": exploratory_png,
    }


def save_final_plot(
    *,
    fig: Any,
    repo_root: str | Path,
    notebook_file: str | Path,
    plot_name: str,
    subfolder: str | None = None,
) -> dict[str, Path]:
    """Save a publication figure as a vector PDF under ``figures/<focus>/final/``.

    Companion to :func:`save_plot` (exploratory PNGs). Relies on the rcParams set
    by :func:`set_publication_style` for font embedding (``pdf.fonttype=42``) and
    tight bbox. Rasterize heavy heatmap meshes in the figure itself
    (``rasterized=True``) so only the cell colour is raster while text and axes
    stay vector.

    The output path is:
    - Without subfolder: ``analysis/figures/<focus>/final/<name>.pdf``
    - With subfolder:    ``analysis/figures/<focus>/final/<subfolder>/<name>.pdf``

    Args:
        fig: Matplotlib Figure object (built at final physical width).
        repo_root: Repository root path containing ``analysis/``.
        notebook_file: Notebook path or filename (sets the focus folder).
        plot_name: Human-readable plot name.
        subfolder: Optional sub-directory appended inside ``final/``.

    Returns:
        Dict with output path under key ``final_pdf``.

    Usage (notebook):
        set_publication_style()
        saved = save_final_plot(fig=fig, repo_root=REPO_ROOT,
                                notebook_file=NOTEBOOK_FILE, plot_name="leaderboard")
    """

    final_dir = _figure_dir(repo_root, notebook_file, stage="final", subfolder=subfolder)
    final_pdf = final_dir / f"{plot_name_slug(plot_name)}.pdf"

    fig.savefig(final_pdf, format="pdf", bbox_inches="tight", pad_inches=0.01)

    return {
        "final_pdf": final_pdf,
    }


def render_de_delta_pair(
    draw_de: Callable[[plt.Axes], Any],
    draw_delta: Callable[[plt.Axes], Any],
    *,
    panel_size: tuple[float, float] = (7.0, 3.4),
    composite_size: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Render a DE panel and its NL−DE delta panel three ways.

    Each callback draws into a single ``Axes`` it is handed. The pair is drawn:
    (1) into its own standalone figure — meant to be written to two separate
    files via :func:`save_panels` — and (2) together into a 1×2 composite figure
    for on-screen side-by-side viewing only. The standardised pattern keeps every
    "Germany + NL−DE delta" cell consistent: two clean panel files, one composite
    to eyeball.

    Args:
        draw_de: Callback ``(ax) -> Any`` rendering the Germany (DE) panel.
        draw_delta: Callback ``(ax) -> Any`` rendering the NL−DE delta panel.
        panel_size: ``(width, height)`` in inches for each standalone panel.
        composite_size: ``(width, height)`` for the 1×2 composite; defaults to
            ``(2 * panel_width, panel_height)``.

    Returns:
        ``(fig_de, fig_delta, fig_composite)``. The caller saves the first two
        (e.g. via :func:`save_panels`), displays the composite, and closes all.
    """
    fig_de, ax_de = plt.subplots(figsize=panel_size)
    draw_de(ax_de)
    fig_de.tight_layout()

    fig_delta, ax_delta = plt.subplots(figsize=panel_size)
    draw_delta(ax_delta)
    fig_delta.tight_layout()

    panel_width, panel_height = panel_size
    fig_composite, (comp_de, comp_delta) = plt.subplots(
        1, 2, figsize=composite_size or (2.0 * panel_width, panel_height)
    )
    draw_de(comp_de)
    draw_delta(comp_delta)
    fig_composite.tight_layout()

    return fig_de, fig_delta, fig_composite


def save_panels(
    *,
    fig_de: Any,
    fig_delta: Any,
    repo_root: str | Path,
    notebook_file: str | Path,
    plot_name: str,
    subfolder: str | None = None,
    dpi: int = 300,
) -> dict[str, Path]:
    """Save a DE panel and its NL−DE delta panel as two separate files.

    Writes ``<plot_name>_de.png`` and ``<plot_name>_delta.png`` into the same
    exploratory folder (and optional ``subfolder``) used by :func:`save_plot`.
    Pairs with :func:`render_de_delta_pair`.

    Args:
        fig_de: The standalone Germany (DE) panel figure.
        fig_delta: The standalone NL−DE delta panel figure.
        repo_root: Repository root path containing ``analysis/``.
        notebook_file: Notebook path or filename (sets the focus folder).
        plot_name: Base name; ``_de`` / ``_delta`` suffixes are appended.
        subfolder: Optional sub-directory inside ``exploratory/``.
        dpi: PNG DPI for exploratory output.

    Returns:
        Dict with keys ``de_png`` and ``delta_png`` → the two output paths.
    """
    de_saved = save_plot(
        fig=fig_de,
        repo_root=repo_root,
        notebook_file=notebook_file,
        plot_name=f"{plot_name}_de",
        dpi=dpi,
        subfolder=subfolder,
    )
    delta_saved = save_plot(
        fig=fig_delta,
        repo_root=repo_root,
        notebook_file=notebook_file,
        plot_name=f"{plot_name}_delta",
        dpi=dpi,
        subfolder=subfolder,
    )
    return {
        "de_png": de_saved["exploratory_png"],
        "delta_png": delta_saved["exploratory_png"],
    }


def plot_emi_leaderboard(
    leaderboard: pd.DataFrame,
    *,
    country: str,
    names: DisplayNames | None = None,
    colors: dict[str, str] | None = None,
    score_col: str = "benchmark_score",
    sd_col: str | None = "ie_sd",
    flag_col: str | None = "rubric_spread",
    flag_threshold: float = 0.20,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Horizontal EMI leaderboard: models ranked by benchmark score.

    Bars are coloured per model, ranked descending (best on top), and labelled
    with integer percentages inside the bar. An optional across-condition SD
    whisker shows the spread that averages into each bar; bars whose ``flag_col``
    reaches ``flag_threshold`` get a high-contrast border — the single headline
    score hides uneven rubric performance (drill deeper in the rubric figure).

    Args:
        leaderboard: Per-(model, country) table with ``model``, ``country`` and
            ``score_col``; optionally ``sd_col`` and ``flag_col``.
        country: Country code to filter to (e.g. ``"DE"``).
        names: Display-name registry; defaults to an empty registry (raw IDs).
        colors: Mapping model *display label* → hex colour (e.g.
            ``model_palette()``). Falls back to a single neutral colour.
        score_col: Column holding the benchmark score in [0, 1].
        sd_col: Column for the descriptive SD whisker, or ``None`` to omit.
        flag_col: Column for the rubric-spread flag, or ``None`` to omit.
        flag_threshold: Flag bars whose ``flag_col`` ≥ this value.
        title: Optional short title (the caption carries the detail).
        ax: Existing Axes to draw into; a new column-width figure is created when
            ``None``.

    Returns:
        The Matplotlib Figure.

    Usage (notebook):
        set_publication_style()
        fig = plot_emi_leaderboard(leaderboard_plot, country="DE",
                                   names=NAMES, colors=model_palette(NAMES))
        save_final_plot(fig=fig, repo_root=REPO_ROOT,
                        notebook_file=NOTEBOOK_FILE, plot_name="emi_leaderboard")
    """

    registry = names or DisplayNames()
    sub = leaderboard[leaderboard["country"] == country].copy()
    sub["model_display"] = sub["model"].map(lambda raw: registry.apply(raw, "model"))
    sub = sub.sort_values(score_col, ascending=False).reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.5 * len(sub) + 0.5))
    else:
        fig = ax.figure

    # Best on top: highest y for the first (best) row.
    y_pos = list(range(len(sub) - 1, -1, -1))
    palette = colors or {}
    bar_colors = [palette.get(m, "#6E6E6E") for m in sub["model_display"]]

    has_flag = bool(flag_col) and flag_col in sub.columns
    flags = (sub[flag_col] >= flag_threshold).tolist() if has_flag else [False] * len(sub)
    edge_colors = [FLAG_EDGE_COLOR if f else "none" for f in flags]
    edge_widths = [0.9 if f else 0.0 for f in flags]

    ax.barh(
        y_pos, sub[score_col], height=0.62, color=bar_colors,
        edgecolor=edge_colors, linewidth=edge_widths, zorder=2,
    )

    has_sd = bool(sd_col) and sd_col in sub.columns
    if has_sd:
        ax.errorbar(
            sub[score_col], y_pos, xerr=sub[sd_col].fillna(0.0), fmt="none",
            ecolor="#3A3A3A", elinewidth=0.8, capsize=2.5, zorder=3,
        )

    for yi, score in zip(y_pos, sub[score_col], strict=True):
        ax.text(
            score - 0.015, yi, f"{round(score * 100)}%", va="center", ha="right",
            fontsize=7, fontweight="bold", color="white", zorder=4,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["model_display"])
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Epistemic Modesty Index")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    ax.margins(y=0.18)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def _style_adherence_cbar(cbar: Any) -> None:
    """Apply the canonical adherence colourbar styling (label + science ticks).

    One definition so every adherence figure's colourbar reads identically to the
    Fig. 1 model×rubric leaderboard.
    """
    cbar.set_label("Adherence")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(direction="in", length=2, width=0.5)
    # Fixed 0.1 ticks so every adherence colourbar reads the same regardless of
    # its rendered height (a tall bar would otherwise auto-tick at 0.05 steps).
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))


def _style_spread_cbar(cbar: Any) -> None:
    """Canonical colourbar styling for the party-spread (YlOrRd) heatmap.

    Mirrors :func:`_style_adherence_cbar` (thin outline, inward ticks) but
    formats the axis as a percent so the spread value reads as 38% rather than 0.38.
    """
    cbar.set_label("Party spread")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(direction="in", length=2, width=0.5)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def _flag_cell_borders(
    ax: plt.Axes,
    flag_matrix: pd.DataFrame,
    *,
    color: str = FLAG_EDGE_COLOR,
    lw: float = 1,
) -> None:
    """Draw a rectangular border around every flagged cell in a seaborn heatmap.

    Seaborn heatmaps don't support per-cell edge colours, so we overlay
    ``matplotlib.patches.Rectangle`` patches.  The patch coordinate system
    matches seaborn's: column ``j`` occupies ``[j, j+1]`` on the x-axis and row
    ``i`` occupies ``[i, i+1]`` on the y-axis (top row = y=0).

    Args:
        ax: The heatmap axes (must be drawn before calling this).
        flag_matrix: Boolean (or 0/1 int) DataFrame with the same row/column
            index as the heatmap pivot — ``True`` / ``1`` marks a cell to border.
        color: Border colour (defaults to :data:`FLAG_EDGE_COLOR`).
        lw: Border line width in points.
    """
    import matplotlib.patches as mpatches
    # δ insets the border so it slightly overlaps cell edges.
    d = 0.03
    for i, row in enumerate(flag_matrix.index):
        for j, col in enumerate(flag_matrix.columns):
            if flag_matrix.loc[row, col]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j + d, i + d), 1 - 2*d, 1 - 2*d,
                        fill=False, edgecolor=color, linewidth=lw, zorder=5
                    )
                )
         


# ── Model logo helpers ────────────────────────────────────────────────────────
# cairosvg and PIL are optional at import time; imported lazily inside the
# helpers so the rest of the module works without them.

# Raw model ID → SVG filename stem in analysis/assets/logos/models/
_MODEL_LOGO_FILE: dict[str, str] = {
    "claude_sonnet_4_6": "claude",
    "gpt_5_4":           "gpt",
    "qwen3_6_flash":     "qwen",
}

# Module-level cache: loaded on first call, keyed by (model_id, source_px)
_LOGO_CACHE: dict[tuple[str, int], Any] = {}

_LOGO_SOURCE_PX: int = 256  # rasterise at this resolution; zoom scales display size


def _load_logo_array(model_id: str) -> Any:
    """Return a cropped RGBA numpy array for *model_id*, cached after first load."""
    import io as _io

    import cairosvg as _cairo
    import numpy as _np
    from PIL import Image as _Image

    key = (model_id, _LOGO_SOURCE_PX)
    if key in _LOGO_CACHE:
        return _LOGO_CACHE[key]

    logo_dir = Path(__file__).resolve().parents[2] / "analysis" / "assets" / "logos" / "models"
    stem = _MODEL_LOGO_FILE.get(model_id, model_id)
    png = _cairo.svg2png(url=str(logo_dir / f"{stem}.svg"),
                         output_width=_LOGO_SOURCE_PX, output_height=_LOGO_SOURCE_PX)
    arr = _np.asarray(_Image.open(_io.BytesIO(png)).convert("RGBA"))

    # Crop to content bounding box so all logos occupy the same apparent area
    # at the same zoom level (Qwen ships with ~10 px transparent margin).
    alpha = arr[:, :, 3]
    rows = _np.any(alpha > 10, axis=1)
    cols = _np.any(alpha > 10, axis=0)
    r0, r1 = _np.where(rows)[0][[0, -1]]
    c0, c1 = _np.where(cols)[0][[0, -1]]
    cropped = arr[r0: r1 + 1, c0: c1 + 1]

    _LOGO_CACHE[key] = cropped
    return cropped


def add_model_logo_ticks(
    ax: plt.Axes,
    model_order: list[str],
    names: Any,                   # DisplayNames
    *,
    logo_pt: float = 13.0,
    right_gap: float = 5.0,
    sep: float = 5.0,
    text_fs: float = 5.5,
    show_label: bool = True,
) -> None:
    """Replace y-axis tick labels with logo-above-label units (Variant B).

    Geometry (points, y measured from row midpoint, positive = up):
        unit_width  = max(logo_pt, widest_label_pt)  — measured at render time
        cx          = -(right_gap + unit_width / 2)  — unit right edge at –right_gap
        logo_cy     = +(sep/2 + text_h/2)            — logo box y-centre
        text_cy     = -(sep/2 + logo_pt/2)           — text y-centre (va=center)

    Only *right_gap* needs to be tweaked per figure: it sets how far the unit's
    right edge sits from the axis spine. Everything else is measured at runtime
    so the helper scales to any font, logo size, or label set.

    Call after ``fig.canvas.draw()`` has been called at least once so the
    renderer is available for text measurement.

    Args:
        ax: Axes whose y-ticks carry model labels (rows = models).
        model_order: Raw model IDs in the display order (top row = index 0).
        names: DisplayNames registry — used to resolve display labels.
        logo_pt: Logo display size in points.
        right_gap: Points between the axis spine and the right edge of the unit.
        sep: Points between logo bottom and label top.
        text_fs: Label font size in points.
        show_label: If False, renders logo only (no text below).
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    labels = [names.apply(m, "model") for m in model_order]
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi

    # Measure the widest label and the common text height at render time.
    max_w_pt = 0.0
    text_h_pt = 0.0
    for lbl in labels:
        t = ax.text(0, 0, lbl, fontsize=text_fs, visible=False)
        bb = t.get_window_extent(renderer)
        max_w_pt  = max(max_w_pt,  bb.width  / dpi * 72)
        text_h_pt = max(text_h_pt, bb.height / dpi * 72)
        t.remove()

    unit_w  = max(logo_pt, max_w_pt) if show_label else logo_pt
    cx      = -(right_gap + unit_w / 2)
    logo_cy =  sep / 2 + text_h_pt / 2    # logo centre: above row midpoint
    text_cy = -(sep / 2 + logo_pt  / 2)   # text centre: below row midpoint

    ax.set_yticks([])
    ax.set_yticklabels([])

    for i, (m, label) in enumerate(zip(model_order, labels, strict=True)):
        arr = _load_logo_array(m)
        zoom = logo_pt / _LOGO_SOURCE_PX
        oi = OffsetImage(arr, zoom=zoom)
        oi.image.axes = ax
        ab = AnnotationBbox(
            oi,
            xy=(0, i + 0.5),
            xycoords=ax.transData,
            xybox=(cx, logo_cy if show_label else 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        # Logos sit left of the data area — disable clipping so they're visible.
        ab.set_clip_on(False)
        ax.add_artist(ab)
        if show_label:
            ax.annotate(
                label,
                xy=(0, i + 0.5),
                xycoords=ax.transData,
                xytext=(cx, text_cy),
                textcoords="offset points",
                ha="center", va="center",
                fontsize=text_fs,
                color="#333333",
            )


# ── Party logo helpers ────────────────────────────────────────────────────────

# Raw party ID → (extension, filename stem) in analysis/assets/logos/parties/
_PARTY_LOGO_FILE: dict[str, tuple[str, str]] = {
    "de_cdu___csu":       ("svg", "de_cdu_csu"),
    "de_spd":             ("svg", "de_spd"),
    "de_grüne":           ("svg", "de_gruene"),
    "de_fdp":             ("svg", "de_fdp"),
    "de_afd":             ("svg", "de_afd"),
    "de_die_linke":       ("svg", "de_linke"),
    "de_bsw":             ("png", "de_bsw"),
    "nl_vvd":             ("svg", "nl_vvd"),
    "nl_d66":             ("svg", "nl_d66"),
    "nl_cda":             ("svg", "nl_cda"),
    "nl_pvv":             ("svg", "nl_pvv"),
    "nl_groenlinks-pvda": ("svg", "nl_gl_pvda"),
    "nl_sp":              ("svg", "nl_sp"),
    "nl_bbb":             ("svg", "nl_bbb"),
    "nl_ja21":            ("svg", "nl_ja21"),
}

_PARTY_LOGO_CACHE: dict[str, Any] = {}
_PARTY_LOGO_SOURCE_PX: int = 256  # SVG rasterisation size; PNG scaled to this height


def _load_party_logo_array(party_id: str) -> Any:
    """Return a cropped RGBA numpy array for *party_id*, cached after first load.

    SVGs are rasterised at a square _PARTY_LOGO_SOURCE_PX canvas. The PNG (BSW)
    is scaled to height=_PARTY_LOGO_SOURCE_PX (aspect preserved) with white
    background removed. All arrays are bbox-cropped so transparent margins don't
    inflate the apparent size.
    """
    import io as _io

    import numpy as _np
    from PIL import Image as _Image

    if party_id in _PARTY_LOGO_CACHE:
        return _PARTY_LOGO_CACHE[party_id]

    if party_id not in _PARTY_LOGO_FILE:
        raise KeyError(f"No logo mapped for party {party_id!r}")

    ext, stem = _PARTY_LOGO_FILE[party_id]
    logo_dir = Path(__file__).resolve().parents[2] / "analysis" / "assets" / "logos" / "parties"
    path = logo_dir / f"{stem}.{ext}"

    if ext == "svg":
        import cairosvg as _cairo

        png = _cairo.svg2png(
            url=str(path),
            output_width=_PARTY_LOGO_SOURCE_PX,
            output_height=_PARTY_LOGO_SOURCE_PX,
        )
        arr = _np.asarray(_Image.open(_io.BytesIO(png)).convert("RGBA"))
    else:
        # PNG (BSW): scale to height, strip white background
        img = _Image.open(path).convert("RGBA")
        scale = _PARTY_LOGO_SOURCE_PX / img.height
        new_w = max(1, int(img.width * scale))
        img = img.resize((new_w, _PARTY_LOGO_SOURCE_PX), _Image.LANCZOS)
        arr = _np.asarray(img).copy()
        white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
        arr[white, 3] = 0

    # Bbox-crop to remove transparent margins
    alpha = arr[:, :, 3]
    rows = _np.any(alpha > 10, axis=1)
    cols = _np.any(alpha > 10, axis=0)
    if rows.any() and cols.any():
        r0, r1 = _np.where(rows)[0][[0, -1]]
        c0, c1 = _np.where(cols)[0][[0, -1]]
        arr = arr[r0 : r1 + 1, c0 : c1 + 1]

    _PARTY_LOGO_CACHE[party_id] = arr
    return arr


def add_party_logo_ticks(
    ax: plt.Axes,
    party_order: list[str],
    names: Any,                    # DisplayNames
    *,
    logo_pt: float = 10.0,
    max_logo_w_pt: float = 22.0,
    right_gap: float = 3.0,
) -> None:
    """Replace y-axis tick labels with party logo icons (logo-only, no text).

    Party logos are wordmarks of varying aspect ratio. Sizing strategy:
    - Effective height per logo = min(logo_pt, max_logo_w_pt / aspect_ratio).
    - Wide logos (e.g. GL-PvdA ratio≈4.74) are capped so their display width
      stays ≤ max_logo_w_pt, making them smaller but recognisable.
    - Square/narrow logos keep the full logo_pt height.
    - The shared horizontal anchor is always max_logo_w_pt so all rows align.

    No text label is rendered — caption handles party identification.

    Call after ``fig.canvas.draw()`` has been called at least once.

    Args:
        ax: Axes whose y-ticks carry party labels (rows = parties).
        party_order: Raw party IDs in display order (top row = index 0).
        names: DisplayNames registry (unused for text here, kept for API parity).
        logo_pt: Maximum logo display height in points.
        max_logo_w_pt: Width budget in points; caps wide logos.
        right_gap: Points between axis spine and right edge of logo.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    # Effective heights: wide logos shrink to fit the width budget
    eff_pts = [
        min(logo_pt, max_logo_w_pt / (_load_party_logo_array(p).shape[1]
                                       / _load_party_logo_array(p).shape[0]))
        for p in party_order
    ]
    cx = -(right_gap + max_logo_w_pt / 2)

    ax.set_yticks([])
    ax.set_yticklabels([])

    for i, (p, eff_pt) in enumerate(zip(party_order, eff_pts, strict=True)):
        arr = _load_party_logo_array(p)
        zoom = eff_pt / arr.shape[0]
        oi = OffsetImage(arr, zoom=zoom)
        oi.image.axes = ax
        ab = AnnotationBbox(
            oi,
            xy=(0, i + 0.5),
            xycoords=ax.transData,
            xybox=(cx, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        # Logos sit outside the axes data area (negative x) — must disable
        # clipping or matplotlib hides them behind the axes bounding box.
        ab.set_clip_on(False)
        ax.add_artist(ab)


def _adherence_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    annot_fontsize: int = ANNOT_FONTSIZE + 1,
    square: bool = True,
    cbar_ax: plt.Axes | None = None,
    vmin: float = ADHERENCE_VMIN,
) -> None:
    """Draw the canonical adherence heatmap onto ``ax``.

    Single source of truth for adherence-matrix styling so every figure (Fig. 1
    leaderboard, the Model×IE overview, the inconclusive small multiples) looks
    identical: fixed viridis 0.60–1.00 scale, integer-percent cell annotations,
    thin white gridlines, and neutral-grey inactive (NaN) cells.

    The caller owns tick labels, titles and figure layout. A colourbar is drawn
    (and styled) only when ``cbar_ax`` is given; otherwise the caller draws it
    (e.g. Fig. 1 pins a manual colourbar to the square-cell block).

    Args:
        ax: Target axes.
        matrix: Models/rows × columns adherence in [0, 1]; NaN = inactive cell.
        annot_fontsize: In-cell percent label size (defaults 1 pt above the
            secondary-annotation base, matching the leaderboard).
        square: Force square cells (True for matrix figures; False when the cell
            geometry is fixed by the axes rectangle, as in the Model×IE overview).
        cbar_ax: Axes to draw the colourbar into; ``None`` to skip it.
    """
    annot = matrix.map(lambda v: f"{round(v * 100)}%" if pd.notna(v) else "")
    mask = matrix.isna().to_numpy()
    # Don't colour the axes background — the Rectangle patches below provide
    # exactly-sized grey squares. A grey facecolor bleeds into the seaborn
    # padding outside the cell grid, making masked columns appear taller.
    sns.heatmap(
        matrix.to_numpy(),
        ax=ax,
        cmap=SEQUENTIAL_CMAP,
        vmin=vmin,
        vmax=ADHERENCE_VMAX,
        mask=mask,
        annot=annot.to_numpy(),
        fmt="",
        annot_kws={"fontsize": annot_fontsize},
        linewidths=0.6,
        linecolor="white",
        square=square,
        cbar=cbar_ax is not None,
        cbar_ax=cbar_ax,
        rasterized=True,
    )
    # Draw masked cells as proper grey squares with white borders matching seaborn's
    # linewidths=0.6 / linecolor="white". Each Rectangle owns its own borders, so
    # grey/grey and grey/colored boundaries look identical to colored/colored ones.
    if mask.any():
        n_rows, n_cols = mask.shape
        for r in range(n_rows):
            for c in range(n_cols):
                if mask[r, c]:
                    ax.add_patch(plt.Rectangle(
                        (c, r), 1, 1,
                        facecolor=INACTIVE_GREY, edgecolor="white",
                        lw=0.6, zorder=3,
                    ))
    if cbar_ax is not None:
        _style_adherence_cbar(ax.collections[0].colorbar)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)


def _delta_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    annot_fontsize: int = ANNOT_FONTSIZE + 1,
    square: bool = True,
    cbar_ax: plt.Axes | None = None,
    vmax: float = 0.15,
) -> None:
    """Draw a NL−DE delta heatmap onto ``ax`` using the diverging colormap.

    Counterpart to :func:`_adherence_heatmap` for country-comparison figures.
    Positive = NL higher; negative = DE higher. NaN cells rendered in grey.

    Args:
        ax: Target axes.
        matrix: Delta values (NL − DE); NaN = inactive cell.
        annot_fontsize: In-cell annotation size.
        square: Force square cells.
        cbar_ax: Axes for the colourbar; ``None`` to skip.
        vmax: Symmetric colour bound (default ±15 pp covers observed range).
    """
    annot = matrix.map(lambda v: f"{v * 100:+.0f}%" if pd.notna(v) else "")
    mask = matrix.isna().to_numpy()
    # Same as _adherence_heatmap: let Rectangle patches provide the grey.
    sns.heatmap(
        matrix.to_numpy(),
        ax=ax,
        cmap=DIVERGING_CMAP,
        vmin=-vmax,
        vmax=vmax,
        center=0,
        mask=mask,
        annot=annot.to_numpy(),
        fmt="",
        annot_kws={"fontsize": annot_fontsize},
        linewidths=0.6,
        linecolor="white",
        square=square,
        cbar=cbar_ax is not None,
        cbar_ax=cbar_ax,
        rasterized=True,
    )
    # Same as _adherence_heatmap: proper grey squares with white borders.
    if mask.any():
        n_rows, n_cols = mask.shape
        for r in range(n_rows):
            for c in range(n_cols):
                if mask[r, c]:
                    ax.add_patch(plt.Rectangle(
                        (c, r), 1, 1,
                        facecolor=INACTIVE_GREY, edgecolor="white",
                        lw=0.6, zorder=3,
                    ))
    if cbar_ax is not None:
        cb = ax.collections[0].colorbar
        cb.set_label("NL − DE")
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(direction="in", length=2, width=0.5)
        cb.ax.yaxis.set_major_locator(MultipleLocator(0.05))
        cb.ax.yaxis.set_major_formatter(_SIGNED_PCT_FORMATTER)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)


def _fmt_delta(v: float, *, threshold: float = 0.005) -> str:
    """Format a signed delta value for bar/text annotations.

    Returns ``"+X%"`` / ``"-X%"`` at integer-percent precision, or ``""`` for
    near-zero values below ``threshold`` (avoids cluttering bars with ``+0%``).
    Centralised so every delta figure uses the same sign+rounding convention.
    """
    if abs(v) < threshold:
        return ""
    return f"{v * 100:+.0f}%"


def _order_ies_by_difficulty(mean_adherence: pd.Series) -> list[str]:
    """Order IEs baseline-first, then by mean adherence descending.

    Baseline is always the leftmost column (the control reference) regardless of
    its score; the remaining IEs follow easiest → hardest so the single colour
    gradient reads as a difficulty ramp.
    """
    rest = [ie for ie in mean_adherence.sort_values(ascending=False).index if ie != "baseline"]
    return (["baseline"] if "baseline" in mean_adherence.index else []) + rest


def plot_model_rubric_leaderboard(
    rubric_table: pd.DataFrame,
    overall: pd.DataFrame,
    *,
    rubric_col: str = "rubric",
    value_col: str = "adherence",
    overall_col: str = "benchmark_score",
    overall_label: str = "EMI",
    col_labels_mode: str = "code",
    title: str | None = None,
    names: DisplayNames | None = None,
    figsize: tuple[float, float] | None = None,
    logo_ticks: bool = False,
    logo_show_label: bool = True,
) -> plt.Figure:
    """Unified model × rubric leaderboard heatmap with a separated EMI column.

    Rows = models ranked by the overall score (best on top); columns = the three
    rubrics in canonical order plus a visually separated ``EMI`` column.
    Colour = adherence on the fixed viridis 0.60–1.00 scale, cells annotated with
    integer percent. Combines the per-rubric diagnostic breakdown and the
    headline ranking in one compact single-column figure with square cells.

    Note: the ``EMI`` column is the geometric across-IE composite, which is
    intentionally *not* the arithmetic mean of the three rubric cells — state
    this in the caption.

    Args:
        rubric_table: Long table with ``model``, ``rubric_col`` and ``value_col``
            for one country (e.g. ``adherence_index(by=["model", "rubric"])``).
        overall: One row per model with ``model`` and ``overall_col`` (e.g.
            ``geometric_benchmark_score``); drives row ranking + Overall column.
        rubric_col: Rubric column name in ``rubric_table``.
        value_col: Adherence column name in ``rubric_table``.
        overall_col: Score column name in ``overall``.
        overall_label: Header for the separated composite column.
        col_labels_mode: Rubric column headers. ``"name"`` = full names on a
            single 45° diagonal; ``"code"`` = compact horizontal F/EC/I codes
            (see ``RUBRIC_CODES``), decoded once in the caption.
        title: Optional short title (the caption carries detail; default None).
        names: Display-name registry (defaults to DISPLAY_NAMES when None).
        figsize: Figure size in inches; defaults to a compact single column.

    Returns:
        The Matplotlib Figure.

    Usage (notebook):
        set_publication_style()
        rub = adherence_index(df[df.country == "DE"], by=["model", "rubric"])
        lb = geometric_benchmark_score(df[df.country == "DE"], by=["model"])
        fig = plot_model_rubric_leaderboard(rub, lb)
    """

    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES
    from src.analysis.display_names import RUBRIC_CODES, RUBRIC_ORDER

    if col_labels_mode not in {"name", "code"}:
        raise ValueError(f"col_labels_mode must be 'name' or 'code', got {col_labels_mode!r}.")

    order = overall.sort_values(overall_col, ascending=False)["model"].tolist()
    overall_map = dict(zip(overall["model"], overall[overall_col], strict=False))

    pivot = rubric_table.pivot_table(index="model", columns=rubric_col, values=value_col)
    rubric_cols = [r for r in RUBRIC_ORDER if r in pivot.columns]
    matrix = pivot.reindex(index=order, columns=rubric_cols)
    matrix[overall_label] = [overall_map.get(m, float("nan")) for m in order]

    if col_labels_mode == "code":
        col_labels = [RUBRIC_CODES.get(r, names.apply(r, "rubric")) for r in rubric_cols]
    else:
        col_labels = [names.apply(r, "rubric") for r in rubric_cols]
    col_labels = col_labels + [overall_label]
    row_labels = [names.apply(m, "model") for m in order]

    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (COLUMN_WIDTH_IN, 0.46 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=figsize)

    # Shared adherence styling; colourbar drawn manually below so its height
    # matches the square cells (square=True shrinks the data region).
    _adherence_heatmap(ax, matrix, annot_fontsize=ANNOT_FONTSIZE + 1, square=True)
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    if col_labels_mode == "code":
        ax.set_xticklabels(col_labels, rotation=0, ha="center")
    else:
        # Single-line 45° anchored: every label ends exactly at its tick, so the
        # long "Epistemic calibration" no longer drifts toward its neighbour.
        ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)

    # Separate the Overall column with a heavy divider (white gap + dark rule).
    ax.axvline(n_cols - 1, color="white", lw=3, zorder=4)
    ax.axvline(n_cols - 1, color="#111111", lw=1.0, zorder=5)

    if title:
        ax.set_title(title)
    fig.tight_layout()

    # Colourbar height must match the *square* cell block, not the axes box.
    # square=True shrinks the data region inside the axes, so a colorbar sized to
    # the axes looks asymmetric. Read the post-draw active position and pin the
    # colorbar to exactly that vertical span.
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes((pos.x1 + 0.02, pos.y0, 0.03, pos.height))
    _style_adherence_cbar(fig.colorbar(ax.collections[0], cax=cax))

    if logo_ticks:
        add_model_logo_ticks(
            ax, order, names,
            show_label=logo_show_label,
        )
    return fig


def plot_model_ie_robustness(
    df: pd.DataFrame,
    *,
    names: DisplayNames | None = None,
    cell_in: float = 0.5,
    target_width: float | None = None,
    mode: str = "level",
    country_a: str = "DE",
    country_b: str = "NL",
    logo_ticks: bool = False,
    logo_show_label: bool = True,
    adherence_vmin: float = ADHERENCE_VMIN,
) -> plt.Figure:
    """Compact Model × IE adherence heatmap with a per-model robustness strip.

    The publication "hero" for §2 (Information Robustness): a square-celled
    adherence matrix (rows = models in rank order, columns = IEs ordered
    baseline-first then easiest → hardest by mean adherence) plus a right-hand
    **Spread (SD)** strip. Cell styling is shared with the Fig. 1 leaderboard
    via :func:`_adherence_heatmap` for one unified look. Designed for a centred
    two-column ``figure*``.

    Two modes:

    - ``"level"`` — single-country adherence (pass ``df`` filtered to one
      country). Right strip = Spread (SD) in brand colours.
    - ``"delta"`` — NL − DE country comparison (pass ``df`` with both
      countries). Matrix shows signed delta using the diverging colormap;
      right strip = NL Spread (SD) only.

    Square cells are guaranteed by absolute-inch axis placement.

    Args:
        df: Tidy frame. For ``mode="level"``, filter to one country first.
            For ``mode="delta"``, pass the full two-country frame.
        names: Display-name registry (defaults to DISPLAY_NAMES when None).
        cell_in: Heatmap cell side length in inches. Ignored when
            ``target_width`` is set (cell size is back-computed from the width).
        target_width: If given, back-compute ``cell_in`` so the figure fills
            exactly this width (e.g. ``TEXT_WIDTH_IN`` for a full-width figure).
            Fixed margins (left, right_pad, gap, rob_w, cbar_w) are preserved;
            only the heatmap cell size changes.
        mode: ``"level"`` or ``"delta"``.
        country_a: Reference country for delta (default ``"DE"``).
        country_b: Comparison country for delta (default ``"NL"``).

    Returns:
        The Matplotlib Figure.

    Usage (notebook 02):
        set_publication_style()
        fig = plot_model_ie_robustness(df[df.country == "DE"])                      # level
        fig = plot_model_ie_robustness(df[df.country == "DE"],
                                       target_width=TEXT_WIDTH_IN)                  # full-width
        fig = plot_model_ie_robustness(df, mode="delta")                            # NL−DE
    """
    if mode not in {"level", "delta"}:
        raise ValueError(f"mode must be 'level' or 'delta', got {mode!r}.")
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES
    from src.analysis.aggregate import adherence_index, robustness_score
    from src.analysis.display_names import MODEL_COLORS, MODEL_ORDER

    if mode == "delta":
        mi_a = adherence_index(df[df["country"] == country_a], by=["model", "ie"]).pivot_table(
            index="model", columns="ie", values="adherence"
        )
        mi_b = adherence_index(df[df["country"] == country_b], by=["model", "ie"]).pivot_table(
            index="model", columns="ie", values="adherence"
        )
        mi_ref = mi_a  # use country_a means to set IE order
        matrix_data = (mi_b - mi_a).reindex(index=MODEL_ORDER)
        # right strip: NL spread only
        rob = (
            robustness_score(df[df["country"] == country_b], by=["model"])
            .set_index("model")["robustness"]
            .reindex(MODEL_ORDER)
        )
    else:
        mi_ref = adherence_index(df, by=["model", "ie"]).pivot_table(
            index="model", columns="ie", values="adherence"
        )
        matrix_data = mi_ref.reindex(index=MODEL_ORDER)
        rob = (
            robustness_score(df, by=["model"]).set_index("model")["robustness"].reindex(MODEL_ORDER)
        )

    order = _order_ies_by_difficulty(mi_ref.mean(axis=0))
    matrix = matrix_data.reindex(columns=order)
    n_col, n_row = matrix.shape[1], matrix.shape[0]

    gap = 0.10
    rob_w, cbar_w = 0.66, 0.14
    left, top, bottom, right_pad = 0.78, 0.30, 0.66, 0.10
    _fixed = left + gap + rob_w + gap + cbar_w + right_pad
    if target_width is not None:
        # Back-compute cell_in so heatmap fills the remaining width exactly.
        cell = max(0.20, (target_width - _fixed) / n_col)
    else:
        cell = cell_in
    heat_w, heat_h = n_col * cell, n_row * cell
    fig_w = left + heat_w + gap + rob_w + gap + cbar_w + right_pad
    fig_h = top + heat_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _rect(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    ax_h = fig.add_axes(_rect(left, bottom, heat_w, heat_h))
    ax_r = fig.add_axes(_rect(left + heat_w + gap, bottom, rob_w, heat_h), sharey=ax_h)
    cax = fig.add_axes(_rect(left + heat_w + gap + rob_w + gap, bottom, cbar_w, heat_h))

    # ── heatmap ──
    if mode == "delta":
        _delta_heatmap(ax_h, matrix, square=False, cbar_ax=cax)
    else:
        _adherence_heatmap(ax_h, matrix, square=False, cbar_ax=cax, vmin=adherence_vmin)
    ax_h.set_yticks([i + 0.5 for i in range(n_row)])
    ax_h.set_yticklabels([names.apply(m, "model") for m in MODEL_ORDER], rotation=0)
    ax_h.set_xticks([i + 0.5 for i in range(n_col)])
    ax_h.set_xticklabels([names.apply(i, "ie") for i in order], rotation=25, ha="right")

    # ── robustness strip: brand-coloured bars on a faint grid ──
    yc = [i + 0.5 for i in range(n_row)]
    ax_r.set_axisbelow(True)
    ax_r.barh(yc, rob.to_numpy(), height=0.6, color=[MODEL_COLORS[m] for m in MODEL_ORDER],
              zorder=3)
    for y, v in zip(yc, rob.to_numpy(), strict=True):
        ax_r.text(v, y, f" {v * 100:.0f}%", va="center", ha="left",
                  fontsize=ANNOT_FONTSIZE, color="#333", zorder=4)
    ax_r.set_ylim(ax_h.get_ylim())
    ax_r.set_xlim(0, float(rob.max()) * 1.75)
    ax_r.xaxis.set_major_locator(MaxNLocator(3))
    ax_r.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_r.grid(axis="x", color="#d7d7d7", lw=0.6)
    ax_r.tick_params(axis="x", labelsize=6, length=0)
    ax_r.tick_params(axis="y", left=False, labelleft=False)
    for spine in ("top", "right", "left"):
        ax_r.spines[spine].set_visible(False)
    ax_r.spines["bottom"].set_color("#bdbdbd")
    _strip_title = "Spread (SD)"
    ax_r.set_title(_strip_title, fontsize=7, pad=4)

    if logo_ticks:
        fig.canvas.draw()
        add_model_logo_ticks(ax_h, MODEL_ORDER, names, show_label=logo_show_label)
    return fig


def plot_ie_rubric_small_multiples(
    df: pd.DataFrame,
    *,
    conditions: Sequence[str] | None = None,
    absent_mode: str = "grey",
    show_scale: bool = True,
    names: DisplayNames | None = None,
    figsize: tuple[float, float] | None = None,
    cell_in: float = 0.5,
    mode: str = "level",
    country_a: str = "DE",
    country_b: str = "NL",
    logo_ticks: bool = False,
    logo_show_label: bool = True,
    show_title: bool = True,
) -> plt.Figure:
    """Per-condition model × rubric heatmaps (adherence or NL−DE delta).

    The §2 supportive figure: one small-multiple panel per condition, each a
    model × rubric adherence heatmap sharing the Fig. 1 styling and a single
    colourbar.

    Two ``mode`` values:

    - ``"level"`` — single-country adherence (pass a single-country ``df``).
    - ``"delta"`` — NL − DE signed delta (pass the two-country ``df``); uses
      the diverging colormap. ``absent_mode`` is honoured the same way.

    Two ``absent_mode`` variants (for Absent / availability):

    - ``"grey"`` — all panels keep all three rubric columns; Absent's inactive
      F/I cells render grey (uniform panel width).
    - ``"collapse"`` — Absent drops empty F/I columns.

    Args:
        df: Tidy frame. For ``mode="level"``, pre-filter to one country.
            For ``mode="delta"``, pass the full two-country frame.
        conditions: IE keys to panel, left → right. Defaults to the
            Inconclusive group; pass ``["baseline", "noise", "prior_conflict"]``
            for the interfering IEs.
        absent_mode: ``"grey"`` or ``"collapse"``.
        show_scale: Draw the shared colourbar.
        names: Display-name registry (defaults to DISPLAY_NAMES when None).
        figsize: Figure size in inches; defaults to content-derived size.
        mode: ``"level"`` or ``"delta"``.
        country_a: Reference country for delta (default ``"DE"``).
        country_b: Comparison country for delta (default ``"NL"``).

    Returns:
        The Matplotlib Figure.

    Usage (notebook 02):
        set_publication_style()
        fig = plot_ie_rubric_small_multiples(df_de)                   # level
        fig = plot_ie_rubric_small_multiples(df, mode="delta")        # NL−DE
        fig = plot_ie_rubric_small_multiples(                         # interfering
            df_de, conditions=["baseline", "noise", "prior_conflict"])
    """
    if absent_mode not in {"grey", "collapse"}:
        raise ValueError(f"absent_mode must be 'grey' or 'collapse', got {absent_mode!r}.")
    if mode not in {"level", "delta"}:
        raise ValueError(f"mode must be 'level' or 'delta', got {mode!r}.")
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES
    from src.analysis.aggregate import adherence_index
    from src.analysis.display_names import IE_ROW_GROUPS, MODEL_ORDER, RUBRIC_CODES, RUBRIC_ORDER

    if conditions is None:
        conditions = next(ies for label, ies in IE_ROW_GROUPS if label == "Inconclusive")

    if mode == "delta":
        rub_a = adherence_index(df[df["country"] == country_a], by=["ie", "model", "rubric"])
        rub_b = adherence_index(df[df["country"] == country_b], by=["ie", "model", "rubric"])
    else:
        rub = adherence_index(df, by=["ie", "model", "rubric"])

    panels: list[tuple[str, pd.DataFrame]] = []
    for cond in conditions:
        if mode == "delta":
            sub_a = (
                rub_a[rub_a["ie"] == cond]
                .pivot_table(index="model", columns="rubric", values="adherence")
                .reindex(index=MODEL_ORDER, columns=RUBRIC_ORDER)
            )
            sub_b = (
                rub_b[rub_b["ie"] == cond]
                .pivot_table(index="model", columns="rubric", values="adherence")
                .reindex(index=MODEL_ORDER, columns=RUBRIC_ORDER)
            )
            sub = sub_b - sub_a
            # restore NaN for cells inactive in both countries
            sub[sub_a.isna() & sub_b.isna()] = float("nan")
        else:
            sub = (
                rub[rub["ie"] == cond]
                .pivot_table(index="model", columns="rubric", values="adherence")
                .reindex(index=MODEL_ORDER, columns=RUBRIC_ORDER)
            )
        if absent_mode == "collapse":
            sub = sub.dropna(axis=1, how="all")
        panels.append((cond, sub))

    n_panels = len(panels)
    n_models = len(MODEL_ORDER)
    panel_cols = [sub.shape[1] for _, sub in panels]
    cbar_ratio = 0.22
    if figsize is None:
        cell = cell_in
        width = 0.95 + sum(panel_cols) * cell + 0.30 * (n_panels - 1) + (0.5 if show_scale else 0.1)
        height = n_models * cell + 1.05
        figsize = (width, height)

    fig = plt.figure(figsize=figsize)
    width_ratios = [*panel_cols, cbar_ratio] if show_scale else panel_cols
    gs = fig.add_gridspec(1, len(width_ratios), width_ratios=width_ratios, wspace=0.18)
    cax = fig.add_subplot(gs[0, n_panels]) if show_scale else None

    first_ax = None
    last_ax = None
    for i, (cond, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if first_ax is None:
            first_ax = ax
        last_ax = ax
        draw_cbar = show_scale and i == n_panels - 1
        if mode == "delta":
            _delta_heatmap(ax, sub, square=True, cbar_ax=cax if draw_cbar else None)
        else:
            _adherence_heatmap(ax, sub, square=True, cbar_ax=cax if draw_cbar else None)
        ax.set_xticks([j + 0.5 for j in range(sub.shape[1])])
        ax.set_xticklabels(
            [RUBRIC_CODES.get(c, names.apply(c, "rubric")) for c in sub.columns],
            rotation=0,
        )
        ax.set_yticks([j + 0.5 for j in range(n_models)])
        ax.set_yticklabels(
            [names.apply(m, "model") for m in MODEL_ORDER] if i == 0 else [""] * n_models,
            rotation=0,
        )
        if show_title:
            ax.set_title(names.apply(cond, "ie"))

    # square=True shrinks each panel's data region; pin the colourbar to the
    # actual cell-block height (post-draw) so it is not taller than the cells.
    if show_scale and cax is not None:
        fig.canvas.draw()
        panel_pos, cbar_pos = last_ax.get_position(), cax.get_position()
        cax.set_position((cbar_pos.x0, panel_pos.y0, cbar_pos.width, panel_pos.height))

    if logo_ticks:
        if not (show_scale and cax is not None):
            fig.canvas.draw()  # ensure renderer is live if not already drawn
        add_model_logo_ticks(first_ax, MODEL_ORDER, names, show_label=logo_show_label)
    return fig


def plot_ie_adherence_overview(
    df: pd.DataFrame,
    *,
    country: str | None = None,
    aggregate_countries: bool = False,
    mode: str = "absolute",
    names: DisplayNames | None = None,
    fig_width: float | None = None,
) -> plt.Figure | list[plt.Figure]:
    """Unified Model × IE adherence overview: heatmap + robustness + difficulty.

    Layout (GridSpec 2×2):
    ┌────────────────────────────┬──────────────────┐
    │  Model × IE heatmap        │  Robustness ↓    │
    │  (viridis, annotated)      │  (horizontal bar)│
    ├────────────────────────────┼──────────────────┤
    │  IE difficulty (bar)       │  colorbar        │
    └────────────────────────────┴──────────────────┘

    Columns ordered baseline-first then easiest → hardest (descending mean
    adherence). Right strip shares row-center positions with the heatmap;
    bottom strip shares column-center positions. The robustness bar fill uses
    the same viridis scale inverted: lower robustness SD → greener (more
    stable); higher → yellower. IE-difficulty bars are viridis-filled by the
    difficulty value itself, keeping visual consistency with the heatmap.

    Country resolution:
    - ``mode="absolute"`` (default):
        - ``aggregate_countries=True``: collapse to equal-country-weight mean
          (average per-country results, never pool raw rows); return a single
          Figure.
        - ``country`` given: filter to that country; single Figure.
        - ``country=None`` + single country in ``df``: existing behavior,
          single Figure.
        - ``country=None`` + multiple countries: render one Figure per country
          (DE first); return a ``list[Figure]``.
    - ``mode="delta"`` (requires exactly 2 countries in ``df``):
        - Incompatible with ``country`` or ``aggregate_countries=True``; raises
          ``ValueError`` if either is set.
        - Heatmap cells = NL−DE delta (diverging cmap; negative = NL
          underperforms). Right robustness strip and bottom IE-difficulty strip
          show NL absolute values.
        - Returns a single Figure.

    Args:
        df: Tidy scores table.
        country: Optional country code to filter to. Incompatible with
            ``aggregate_countries=True`` and ``mode="delta"``.
        aggregate_countries: When ``True`` (absolute mode only), collapse
            multi-country frames to equal-country-weight averages.
        mode: ``"absolute"`` (default) or ``"delta"`` (requires 2 countries).
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        fig_width: Total figure width in inches; defaults to a sensible value
            based on data dimensions.

    Returns:
        A single ``Figure`` or a ``list[Figure]`` (multi-country absolute mode).

    Raises:
        ValueError: If ``mode="delta"`` is used with ``country`` or
            ``aggregate_countries=True``, or if fewer/more than 2 countries are
            present in ``delta`` mode.

    Usage (notebook 02 — §1/§2 Model×IE overview):
        fig = plot_ie_adherence_overview(df_de)                    # DE focus
        fig = plot_ie_adherence_overview(df, mode="delta")         # NL−DE deviation
        fig = plot_ie_adherence_overview(df, aggregate_countries=True)  # combined
        fig = plot_ie_adherence_overview(df, country="DE")         # explicit filter
        save_plot(fig=fig, repo_root=REPO_ROOT, notebook_file=NOTEBOOK_FILE,
                  plot_name="model_ie_heatmap")
    """
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    # ── Guard: mode="delta" incompatibilities ───────────────────────────────
    if mode == "delta":
        if country is not None:
            raise ValueError(
                "mode='delta' is incompatible with a country filter — "
                "remove country= or use mode='absolute'."
            )
        if aggregate_countries:
            raise ValueError("mode='delta' is incompatible with aggregate_countries=True.")
        present = _countries_in(df)
        if len(present) != 2:
            raise ValueError(f"mode='delta' requires exactly 2 countries in df; found {present!r}.")
        return _plot_overview_delta(df, names=names, fig_width=fig_width)

    # ── Resolve countries for absolute mode ─────────────────────────────────
    if aggregate_countries:
        return _plot_overview_absolute(df, names=names, fig_width=fig_width, aggregate=True)

    if country is not None:
        df_filtered = df[df["country"] == country]
        return _plot_overview_absolute(
            df_filtered, names=names, fig_width=fig_width, aggregate=False
        )

    present = _countries_in(df)
    if len(present) <= 1:
        return _plot_overview_absolute(df, names=names, fig_width=fig_width, aggregate=False)

    # Multi-country: render one figure per country (DE first).
    return [
        _plot_overview_absolute(
            df[df["country"] == c], names=names, fig_width=fig_width, aggregate=False
        )
        for c in present
    ]


def plot_ie_rubric_heatmap(
    df: pd.DataFrame,
    *,
    ie: str,
    country: str | None = None,
    show_country_correlation: bool = False,
    flag_threshold: float | None = None,
    names: DisplayNames | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes | list[plt.Axes]:
    """Model x rubric adherence heatmap for a single IE, with SQ-spread flags.

    Cell = adherence by model x active rubric within ``ie``. A trailing ``*`` is
    added to any cell whose sub-question spread (max - min across sub-questions)
    is at least ``flag_threshold`` — the rubric mean hides an uneven
    sub-question there. Inactive rubrics for the IE appear as blank cells.

    Country resolution:
    - ``country`` given → filter ``df`` to it; draw single heatmap; return
      ``Axes``.
    - ``country=None`` + exactly one country in ``df`` → existing behavior;
      return single ``Axes``.
    - ``country=None`` + >1 country → render one heatmap per country (DE
      first), each in its own new figure; return ``list[Axes]``. Raises
      ``ValueError`` if ``ax`` is also provided in this case.

    When ``show_country_correlation=True`` and ≥2 countries are present in the
    *pre-filter* ``df``, the per-rubric model-rank ρ (from
    :func:`~src.analysis.replication.model_rank_agreement` at level ``"rubric"``)
    is appended to each rubric column x-tick label (e.g. ``Faithfulness\nρ=0.50``).
    Annotated on every rendered country panel. If <2 countries are present in
    the original ``df``, the annotation is silently skipped.

    Args:
        df: Tidy scores table (all IEs; filtered internally to ``ie``).
        ie: Raw IE id (e.g. ``"noise"``).
        country: Optional country code. When given, only that country is plotted.
        show_country_correlation: Append per-rubric model-rank ρ to x-tick
            labels when ≥2 countries are in ``df``.
        flag_threshold: SQ-spread flag cutoff; defaults to the module-wide
            ``FLAG_THRESHOLD``.
        names: Display-name registry (defaults to the shared ``DISPLAY_NAMES``).
        ax: Existing axis to draw on; a new figure is created when ``None``.
            Must not be provided in multi-country auto-render mode.

    Returns:
        A single ``Axes`` or a ``list[Axes]`` (multi-country auto-render).

    Raises:
        ValueError: If ``ax`` is provided in multi-country auto-render mode.

    Usage (notebook 02 — §3, single country or auto-both):
        ax = plot_ie_rubric_heatmap(df_de, ie="noise")             # single country
        axes = plot_ie_rubric_heatmap(df, ie="noise")              # auto-both (list)
        axes = plot_ie_rubric_heatmap(df, ie="noise",
                                      show_country_correlation=True)  # + rubric ρ
        # For side-by-side DE + NL−DE delta, drive render_de_delta_pair with a
        # country="DE" heatmap and plot_country_delta_heatmap as the two panels.
    """
    from src.analysis.aggregate import FLAG_THRESHOLD

    if flag_threshold is None:
        flag_threshold = FLAG_THRESHOLD
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    # Keep the original df for show_country_correlation (needs pre-filter data).
    df_all = df

    # ── Country-correlation rho lookup ───────────────────────────────────────
    # Compute once against the full pre-filter df if requested.
    rho_by_rubric_display: dict[str, str] = {}
    if show_country_correlation and len(_countries_in(df_all)) >= 2:
        from src.analysis.replication import model_rank_agreement

        df_ie_all = df_all[df_all["ie"] == ie]
        rank_df = model_rank_agreement(df_ie_all, level="rubric")
        # Map display name → "ρ=0.50" annotation.
        for _, row in rank_df.iterrows():
            raw_rubric = str(row["group"])
            display_rubric = names.apply(raw_rubric, "rubric")
            rho_val = row["rho"]
            if pd.isna(rho_val):
                rho_by_rubric_display[display_rubric] = "ρ=n/a"
            else:
                rho_by_rubric_display[display_rubric] = f"ρ={float(rho_val):.2f}"

    # ── Resolve country ──────────────────────────────────────────────────────
    if country is not None:
        df_filtered = df[df["country"] == country]
        return _draw_ie_rubric_heatmap(
            df_filtered,
            ie=ie,
            flag_threshold=flag_threshold,
            names=names,
            ax=ax,
            rho_by_rubric_display=rho_by_rubric_display,
        )

    present = _countries_in(df)
    if len(present) <= 1:
        return _draw_ie_rubric_heatmap(
            df,
            ie=ie,
            flag_threshold=flag_threshold,
            names=names,
            ax=ax,
            rho_by_rubric_display=rho_by_rubric_display,
        )

    # Multi-country: one panel per country.
    if ax is not None:
        raise ValueError(
            "Cannot draw multiple countries on a single supplied axis. "
            "Either pass country= to select one country or omit ax=."
        )
    return [
        _draw_ie_rubric_heatmap(
            df[df["country"] == c],
            ie=ie,
            flag_threshold=flag_threshold,
            names=names,
            ax=None,
            rho_by_rubric_display=rho_by_rubric_display,
        )
        for c in present
    ]


def plot_country_delta_heatmap(
    df: pd.DataFrame,
    *,
    axis: str,
    ie: str | None = None,
    star_from_nl_sq: bool = False,
    country_a: str = "DE",
    country_b: str = "NL",
    names: DisplayNames | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Generic model × {rubric|ie} NL−DE deviation heatmap (reusable drill primitive).

    Cells show the signed adherence delta (``country_b`` − ``country_a``,
    i.e. NL − DE by default; negative = country_b underperforms) using a
    diverging colormap centered at 0. A trailing ``*`` can be added to flag cells
    where the ``country_b`` aggregate hides lopsided sub-question scores.

    Args:
        df: Tidy scores table.
        axis: Grouping axis for the columns — ``"rubric"`` or ``"ie"``.
        ie: When ``axis="rubric"``, optionally filter ``df`` to a single IE first.
            Ignored when ``axis="ie"``.
        star_from_nl_sq: When ``True`` (only meaningful for ``axis="rubric"``),
            append ``*`` to any cell whose ``country_b`` sub-question spread
            (``sq_spread >= FLAG_THRESHOLD``) flags a lopsided sub-question.
            Silently ignored when ``axis="ie"``.
        country_a: Reference country code (default ``"DE"``); the baseline.
        country_b: Comparison country code (default ``"NL"``); delta = country_b − country_a.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        ax: Existing axis to draw on; a new figure is created when ``None``.

    Returns:
        The matplotlib ``Axes`` the heatmap was drawn on.

    Raises:
        ValueError: If ``axis`` is not ``"rubric"`` or ``"ie"``.

    Usage (notebook 02 — §3 per-IE and §4 replication):
        ax = plot_country_delta_heatmap(df, axis="rubric", ie="noise",
                                        star_from_nl_sq=True)      # per-IE
        ax = plot_country_delta_heatmap(df, axis="ie")             # §4 model×IE
        save_plot(fig=ax.figure, repo_root=REPO_ROOT, notebook_file=NOTEBOOK_FILE,
                  plot_name="country_delta_noise", subfolder="noise")
        # NL SQ breakdown: plot_subquestion_adherence(df, ie="noise",
        #   rubric="impartiality", country="NL", baseline_reference=True)
    """
    from matplotlib.colors import TwoSlopeNorm

    from src.analysis.aggregate import FLAG_THRESHOLD, subquestion_dispersion
    from src.analysis.replication import country_delta

    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    if axis not in ("rubric", "ie"):
        raise ValueError(f"axis must be 'rubric' or 'ie'; got {axis!r}")

    # ── Filter and grouping ──────────────────────────────────────────────────
    df_work = df.copy()
    if axis == "rubric" and ie is not None:
        df_work = df_work[df_work["ie"] == ie]

    by: list[str] = ["model", axis]
    delta_df = country_delta(df_work, by=by, country_a=country_a, country_b=country_b)

    # Build pivot: model (index) × {rubric|ie} (columns) = delta values.
    pivot = delta_df.pivot(index="model", columns=axis, values="delta")
    pivot.columns.name = None

    # Apply display names.
    pivot.index = [names.apply(m, "model") for m in pivot.index]
    pivot.columns = [names.apply(col, axis) for col in pivot.columns]

    # ── SQ-spread star flags (country_b only, axis="rubric") ─────────────────
    star_mask: pd.DataFrame | None = None
    if star_from_nl_sq and axis == "rubric":
        df_b = df_work[df_work["country"] == country_b]
        spread_df = subquestion_dispersion(df_b, by=("model", "country", "rubric"))
        spread_pivot = spread_df.pivot(index="model", columns="rubric", values="sq_spread")
        spread_pivot.columns.name = None
        spread_pivot.index = [names.apply(m, "model") for m in spread_pivot.index]
        spread_pivot.columns = [names.apply(r, "rubric") for r in spread_pivot.columns]
        # Align with delta pivot (same index/columns order).
        star_mask = spread_pivot.reindex(index=pivot.index, columns=pivot.columns)

    # ── Build annotation matrix ───────────────────────────────────────────────
    annot = pivot.copy().astype(object)
    for row in pivot.index:
        for col in pivot.columns:
            val = pivot.loc[row, col]
            if pd.isna(val):
                annot.loc[row, col] = ""
                continue
            cell_str = f"{val:+.2f}"
            if star_mask is not None:
                _in_mask = row in star_mask.index and col in star_mask.columns
                sq_val = star_mask.loc[row, col] if _in_mask else float("nan")
                if pd.notna(sq_val) and float(sq_val) >= FLAG_THRESHOLD:
                    cell_str += "*"
            annot.loc[row, col] = cell_str

    # ── Diverging norm ────────────────────────────────────────────────────────
    heat_vals = pivot.values[~pd.isna(pivot.values)]
    _raw_abs = (
        max(abs(float(heat_vals.min())), abs(float(heat_vals.max()))) if len(heat_vals) else 0.0
    )
    # TwoSlopeNorm requires vmin < vcenter < vmax; guard against all-zero deltas.
    abs_max = _raw_abs if _raw_abs > 1e-9 else 1e-6
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.5 * len(pivot.columns) + 2.0, 0.7 * len(pivot) + 1.6))

    sns.heatmap(
        pivot,
        annot=annot.values,
        fmt="",
        cmap="RdBu_r",
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Δ Adherence ({country_b}−{country_a})"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    axis_label = names.apply(ie, "ie") if (ie is not None and axis == "rubric") else ""
    star_note = "  (* = NL SQ spread flagged)" if star_from_nl_sq and axis == "rubric" else ""
    ax.set_title(
        f"Deviation from {country_a} ({country_b}−{country_a}) by {axis}"
        + (f"  [{axis_label}]" if axis_label else "")
        + star_note,
        pad=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return ax


def plot_subquestion_adherence(
    df: pd.DataFrame,
    *,
    rubric: str,
    ie: str | None = None,
    models: list[str] | None = None,
    country: str | None = None,
    show_average: bool = True,
    show_ie_spread: bool = False,
    baseline_reference: bool = False,
    names: DisplayNames | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Per-sub-question pass-rate bars for a rubric (the drill).

    Filters to ``rubric`` (and to ``ie`` when given, else pools across all
    conditions), optionally to ``models`` / ``country``, then plots each
    sub-question's pass rate as a horizontal bar, ascending so the worst
    sub-question sits at the bottom. Bars are grouped by model.

    When ``show_average`` is set, a grey model-average bar is added at the bottom
    of each sub-question group. When ``baseline_reference`` is set and a specific
    non-baseline ``ie`` is shown, a red tick marks each sub-question's
    model-averaged *baseline* pass rate on that average bar — but only for
    sub-questions that are also active at baseline (e.g. ``structural_balance``,
    active only under Contradiction, gets no tick).

    When ``show_ie_spread`` is set and ``ie`` is ``None`` (pooled across all
    conditions), a dark whisker spanning the min–max of per-IE pass rates is
    overlaid on every bar — showing how stable each sub-question score is across
    information environment conditions. Ignored when a specific ``ie`` is given.

    Args:
        df: Tidy scores table.
        rubric: Raw rubric id to inspect (e.g. ``"impartiality"``).
        ie: Raw IE id to inspect (e.g. ``"noise"``); ``None`` pools all IEs.
        models: Optional subset of raw model ids; all models when ``None``.
        country: Optional single country code; all countries when ``None``.
        show_average: Add a model-average bar per sub-question.
        show_ie_spread: Overlay per-IE min–max whiskers on each bar (only when
            ``ie`` is ``None``; silently ignored otherwise).
        baseline_reference: Mark the baseline pass rate on the average bar
            (ignored unless a specific non-baseline ``ie`` is shown).
        names: Display-name registry (defaults to the shared ``DISPLAY_NAMES``).
        ax: Existing axis to draw on; a new figure is created when ``None``.
        title: Override title; a sensible default is built when ``None``.

    Returns:
        The matplotlib ``Axes`` the bars were drawn on.

    Usage (notebook 01 — §3 SQ breakdown; notebook 02 — drill cells):
        ax = plot_subquestion_adherence(df, rubric="impartiality")  # nb01 overall
        ax = plot_subquestion_adherence(df_de, ie="noise",
                                        rubric="impartiality",
                                        baseline_reference=True)    # nb02 drill
        ax = plot_subquestion_adherence(df, ie="noise",             # NL drill
                                        rubric="impartiality",
                                        country="NL",
                                        baseline_reference=True)
        save_plot(fig=ax.figure, repo_root=REPO_ROOT, notebook_file=NOTEBOOK_FILE,
                  plot_name="sq_noise_impartiality")
    """
    from matplotlib.lines import Line2D

    from src.analysis.aggregate import subquestion_adherence

    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    # Build filtered frame once — shared by pass-rate computation and IE spread.
    _base_frame = df[df["rubric"] == rubric]
    if country is not None:
        _base_frame = _base_frame[_base_frame["country"] == country]
    if models is not None:
        _base_frame = _base_frame[_base_frame["model"].isin(models)]

    def _sq_rates(ie_value: str | None) -> pd.DataFrame:
        frame = _base_frame if ie_value is None else _base_frame[_base_frame["ie"] == ie_value]
        out = subquestion_adherence(frame, by=("model", "subquestion"))
        out["subquestion_label"] = out["subquestion"].str.replace("_", " ")
        return out

    rates = _sq_rates(ie)
    rates["series"] = rates["model"].map(lambda m: names.apply(m, "model"))

    model_series = rates["series"].drop_duplicates().tolist()
    plot_df = rates[["subquestion_label", "series", "adherence"]]
    if show_average:
        avg = (
            rates.groupby("subquestion_label", as_index=False)["adherence"]
            .mean()
            .assign(series="Average")
        )
        plot_df = pd.concat(
            [plot_df, avg[["subquestion_label", "series", "adherence"]]], ignore_index=True
        )

    order = rates.groupby("subquestion_label")["adherence"].mean().sort_values().index.tolist()
    hue_order = model_series + (["Average"] if show_average else [])
    base_colors = sns.color_palette()
    palette = {m: base_colors[i] for i, m in enumerate(model_series)}
    if show_average:
        palette["Average"] = "0.6"

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.62 * len(order) + 1.6))
    sns.barplot(
        data=plot_df,
        x="adherence",
        y="subquestion_label",
        hue="series",
        order=order,
        hue_order=hue_order,
        palette=palette,
        orient="h",
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=7)

    # --- IE-spread whiskers: min–max of per-IE pass rates across all conditions ---
    # Only meaningful when pooling across IEs (ie=None); skip for single-IE views.
    _whisker_drawn = False
    if show_ie_spread and ie is None:
        per_ie = subquestion_adherence(_base_frame, by=("model", "subquestion", "ie"))
        per_ie["subquestion_label"] = per_ie["subquestion"].str.replace("_", " ")
        ie_range = (
            per_ie.groupby(["model", "subquestion_label"])["adherence"]
            .agg(ie_min="min", ie_max="max")
            .reset_index()
        )
        ie_range["series"] = ie_range["model"].map(lambda m: names.apply(m, "model"))

        avg_ie_range: pd.DataFrame | None = None
        if show_average:
            # Average across models per (subquestion, ie) first, then take min/max across IEs.
            avg_per_ie = (
                per_ie.groupby(["subquestion_label", "ie"])["adherence"].mean().reset_index()
            )
            avg_ie_range = (
                avg_per_ie.groupby("subquestion_label")["adherence"]
                .agg(ie_min="min", ie_max="max")
                .reset_index()
            )

        _WHISKER_COLOR = "#444444"
        _CAP_FRAC = 0.3  # fraction of bar height used for end caps

        for series_name, container in zip(hue_order, ax.containers, strict=True):
            for sq_label, patch in zip(order, container, strict=True):
                if series_name == "Average":
                    if avg_ie_range is None:
                        continue
                    row = avg_ie_range[avg_ie_range["subquestion_label"] == sq_label]
                else:
                    row = ie_range[
                        (ie_range["series"] == series_name)
                        & (ie_range["subquestion_label"] == sq_label)
                    ]
                if row.empty:
                    continue
                x_min = float(row["ie_min"].iloc[0])
                x_max = float(row["ie_max"].iloc[0])
                y_mid = patch.get_y() + patch.get_height() / 2
                cap_h = patch.get_height() * _CAP_FRAC

                ax.plot(
                    [x_min, x_max],
                    [y_mid, y_mid],
                    color=_WHISKER_COLOR,
                    lw=1.2,
                    solid_capstyle="butt",
                    zorder=3,
                )
                ax.plot(
                    [x_min, x_min],
                    [y_mid - cap_h / 2, y_mid + cap_h / 2],
                    color=_WHISKER_COLOR,
                    lw=1.2,
                    zorder=3,
                )
                ax.plot(
                    [x_max, x_max],
                    [y_mid - cap_h / 2, y_mid + cap_h / 2],
                    color=_WHISKER_COLOR,
                    lw=1.2,
                    zorder=3,
                )
        _whisker_drawn = True

    # Baseline tick on the average bar, only for sub-questions active at baseline.
    show_baseline = baseline_reference and show_average and ie is not None and ie != "baseline"
    if show_baseline:
        baseline_by_sq = _sq_rates("baseline").groupby("subquestion_label")["adherence"].mean()
        avg_container = ax.containers[hue_order.index("Average")]
        for label, patch in zip(order, avg_container, strict=True):
            value = baseline_by_sq.get(label)
            if value is None or pd.isna(value):
                continue
            ax.plot(
                [value, value],
                [patch.get_y(), patch.get_y() + patch.get_height()],
                color="#B22222",
                lw=2.0,
                solid_capstyle="butt",
                zorder=5,
            )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Sub-question pass rate")
    ax.set_ylabel("")
    ie_label = names.apply(ie, "ie") if ie is not None else "All conditions"
    default_title = f"{ie_label} x {names.apply(rubric, 'rubric')} — sub-question adherence"
    ax.set_title(title or default_title, pad=8)

    handles, labels = ax.get_legend_handles_labels()
    if show_baseline:
        handles.append(Line2D([0], [0], color="#B22222", lw=2.0))
        labels.append("Baseline (avg)")
    if _whisker_drawn:
        handles.append(Line2D([0], [0], color="#444444", lw=1.2))
        labels.append("IE range (min–max)")
        ax.annotate(
            "Whiskers = per-IE pass-rate range (min–max across all conditions)",
            xy=(0.01, 0.01),
            xycoords="axes fraction",
            fontsize=6,
            color="#666666",
            va="bottom",
        )
    ax.legend(handles, labels, title="", frameon=False, loc="lower right")
    sns.despine(ax=ax)
    return ax


def plot_subquestion_pair(
    df: pd.DataFrame,
    rubric: str,
    *,
    country_a: str = "DE",
    country_b: str = "NL",
    exclude: tuple[str, ...] = (),
    names: DisplayNames | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Two-panel per-sub-question breakdown for one rubric (publication/appendix).

    (a) ``country_a`` sub-question pass rates per model; (b) the
    ``country_b`` − ``country_a`` delta per model. Both panels share one
    y-order — sub-questions ascending by ``country_a`` mean, so the weakest sits
    at the top. Models use the fixed brand palette; there is no model-average
    bar (irrelevant for the per-model story). A single shared legend sits below
    the panels so it never overlays the bars. Panel titles are just ``(a)``/``(b)``
    — the descriptive detail belongs in the caption.

    Args:
        df: Tidy scores frame spanning both countries.
        rubric: Rubric id to break down (e.g. ``"impartiality"``).
        country_a: Reference country — absolute pass rates (panel a).
        country_b: Comparison country — delta = b − a (panel b).
        exclude: Sub-question ids to drop (e.g. ``("structural_balance",)``).
        names: Display-name registry (defaults to DISPLAY_NAMES).
        figsize: Figure size in inches; defaults to full text width.

    Returns:
        The Matplotlib Figure.

    Usage (notebook):
        set_publication_style()
        fig = plot_subquestion_pair(df, "impartiality", exclude=("structural_balance",))
        save_final_plot(fig=fig, repo_root=REPO_ROOT, notebook_file=NOTEBOOK_FILE,
                        plot_name="sq_impartiality_pair")
    """
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES
    from src.analysis.aggregate import subquestion_adherence
    from src.analysis.display_names import MODEL_ORDER, model_palette

    frame = df[df["rubric"] == rubric]
    if exclude:
        frame = frame[~frame["subquestion"].isin(exclude)]

    rates = subquestion_adherence(frame, by=("model", "subquestion", "country"))
    wide = (
        rates.pivot_table(index=["model", "subquestion"], columns="country", values="adherence")
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.dropna(subset=[country_a, country_b])
    wide["delta"] = wide[country_b] - wide[country_a]
    wide["model_display"] = wide["model"].map(lambda m: names.apply(m, "model"))

    # SQ labels with the code at the *end* — "Endorsement (I1)" reads cleaner than
    # the registry's leading "I1 · Endorsement" on a y-axis.
    def _code_last(raw: str) -> str:
        label = names.apply(raw, "subquestion")
        if " · " in label:
            code, name = label.split(" · ", 1)
            return f"{name} ({code})"
        return label

    wide["sq_display"] = wide["subquestion"].map(_code_last)

    # Shared y-order: weakest SQ at the top (seaborn puts the first entry at the
    # top, so sort ascending by country_a mean) — surfaces Sanitization first.
    sq_order = wide.groupby("sq_display")[country_a].mean().sort_values().index.tolist()
    present = set(wide["model"])
    model_order = [names.apply(m, "model") for m in MODEL_ORDER if m in present]
    palette = model_palette(names)

    # Full text width, slim bars (width=0.7 + gaps). Fonts ~10% below the global
    # publication sizes — this dense two-panel breakdown reads better small.
    tick_fs, label_fs, title_fs, bar_fs, legend_fs = 7, 8, 8, 5.5, 7
    if figsize is None:
        figsize = (TEXT_WIDTH_IN, 0.38 * len(sq_order) + 1.2)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    common = {"y": "sq_display", "hue": "model_display", "order": sq_order,
              "hue_order": model_order, "palette": palette, "orient": "h",
              "width": 0.7, "gap": 0.1}
    ceiling = 0.96  # at/above this, the label goes *inside* (white) to avoid clipping

    # (a) absolute pass rates: labels outside the bar, except ceiling bars (inside).
    sns.barplot(data=wide, x=country_a, ax=ax_a, **common)
    for container in ax_a.containers:
        for patch, val in zip(container, container.datavalues, strict=True):
            pct = round(val * 100)
            if pct >= 99:
                continue  # full bars need no label
            ymid = patch.get_y() + patch.get_height() / 2
            inside = val >= ceiling
            ax_a.text(
                val + (-0.012 if inside else 0.012), ymid, f"{pct}%",
                ha="right" if inside else "left", va="center",
                color="white" if inside else "#333333", fontsize=bar_fs,
            )
    ax_a.set_xlim(0, 1)
    ax_a.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_a.set_xlabel(f"({country_a}) pass rate", fontsize=label_fs)
    ax_a.set_ylabel("")
    ax_a.set_title("(a)", loc="left", fontsize=title_fs)
    ax_a.tick_params(labelsize=tick_fs)
    if ax_a.get_legend() is not None:
        ax_a.get_legend().remove()
    # Pin y-limits to exact group range so the seaborn categorical separator
    # lines don't repeat just inside the top/bottom spine.
    n_sq = len(sq_order)
    ax_a.set_ylim(n_sq - 0.5, -0.5)
    sns.despine(ax=ax_a)

    # (b) cross-country delta — label each bar with sign, skip near-zero.
    sns.barplot(data=wide, x="delta", ax=ax_b, **common)
    for container in ax_b.containers:
        labels = [_fmt_delta(v) for v in container.datavalues]
        ax_b.bar_label(container, labels=labels, padding=2,
                       fontsize=bar_fs, color="#333333")
    # Round up to nearest 5 pp so tick edges land at clean values and don't
    # crowd the axis boundary — avoids the ghost-separator artefact.
    raw_lim = float(wide["delta"].abs().max()) * 1.5 + 1e-6
    lim = math.ceil(raw_lim * 20) / 20  # ceil to nearest 0.05
    ax_b.set_xlim(-lim, lim)
    ax_b.xaxis.set_major_formatter(_SIGNED_PCT_FORMATTER)
    ax_b.axvline(0, color="#333333", lw=0.8, zorder=2)
    ax_b.set_xlabel(f"{country_b} − {country_a}  (Δ pass rate)", fontsize=label_fs)
    ax_b.set_ylabel("")
    ax_b.set_title("(b)", loc="left", fontsize=title_fs)
    ax_b.set_ylim(n_sq - 0.5, -0.5)
    ax_b.tick_params(labelleft=False, labelsize=tick_fs)
    if ax_b.get_legend() is not None:
        ax_b.get_legend().remove()
    sns.despine(ax=ax_b)

    # Shared legend in a reserved bottom band — cleanly separated from the x-labels.
    handles, labels = ax_a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.legend(
        handles, labels, ncol=len(labels), frameon=False, fontsize=legend_fs,
        loc="lower center", bbox_to_anchor=(0.5, 0.0), columnspacing=1.6,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def plot_party_adherence_heatmap(
    df: pd.DataFrame,
    *,
    ie: str = "baseline",
    country: str | None = None,
    flag_threshold: float | None = None,
    names: DisplayNames | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Model × party adherence heatmap for a single IE, with rubric-spread flags.

    Cell = adherence by model × party within ``ie``. A trailing ``*`` flags any
    cell whose rubric spread (max − min across active rubrics for that model ×
    party) is at least ``flag_threshold`` — the composite mean hides uneven
    rubric coverage there.

    Columns are ordered by ``PARTY_ORDER`` (display names); unknown parties fall
    to the right in alphabetical order.

    Args:
        df: Tidy scores table.
        ie: Raw IE id (default ``"baseline"``).
        country: Optional country code; filters ``df`` when given.
        flag_threshold: Rubric-spread flag cutoff; defaults to ``FLAG_THRESHOLD``.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        ax: Existing axis; a new figure is created when ``None``.

    Returns:
        The matplotlib ``Axes`` the heatmap was drawn on.
    """
    from src.analysis.aggregate import FLAG_THRESHOLD, adherence_index, rubric_dispersion
    from src.analysis.display_names import PARTY_ORDER

    if flag_threshold is None:
        flag_threshold = FLAG_THRESHOLD
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    df_ie = df[df["ie"] == ie]
    if country is not None:
        df_ie = df_ie[df_ie["country"] == country]

    adh = adherence_index(df_ie, by=["model", "country", "party"])
    rdisp = rubric_dispersion(df_ie, by=["model", "country", "party"])
    merged = adh.merge(rdisp, on=["model", "country", "party"], how="left")
    merged["model_display"] = merged["model"].map(lambda m: names.apply(m, "model"))
    merged["party_display"] = merged["party"].map(lambda p: names.apply(p, "party"))

    adh_pivot = merged.pivot_table(
        index="model_display", columns="party_display", values="adherence"
    )
    spread_pivot = merged.pivot_table(
        index="model_display", columns="party_display", values="rubric_spread"
    )

    # Order columns by PARTY_ORDER (display names), unknown parties at end.
    party_display_order = [names.apply(p, "party") for p in PARTY_ORDER]
    ordered_cols = [c for c in party_display_order if c in adh_pivot.columns]
    ordered_cols += sorted(c for c in adh_pivot.columns if c not in ordered_cols)
    adh_pivot = adh_pivot[ordered_cols]
    spread_pivot = spread_pivot[ordered_cols]

    annot = adh_pivot.copy().astype(object)
    for row in adh_pivot.index:
        for col in adh_pivot.columns:
            value = adh_pivot.loc[row, col]
            if pd.isna(value):
                annot.loc[row, col] = ""
                continue
            rspread = spread_pivot.loc[row, col]
            flagged = pd.notna(rspread) and rspread >= flag_threshold
            annot.loc[row, col] = f"{value:.2f}" + ("*" if flagged else "")

    if ax is None:
        _, ax = plt.subplots(
            figsize=(1.1 * len(adh_pivot.columns) + 2.0, 0.7 * len(adh_pivot) + 1.6)
        )
    ie_label = names.apply(ie, "ie")
    country_label = f" ({country})" if country else ""
    sns.heatmap(
        adh_pivot,
        annot=annot.values,
        fmt="",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Adherence index"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{ie_label}{country_label} — model × party  (* = rubric spread >= {flag_threshold:.2f})",
        pad=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return ax


def plot_party_spread_heatmap(
    df: pd.DataFrame,
    *,
    country: str | None = None,
    flag_threshold: float | None = None,
    names: DisplayNames | None = None,
    vmax: float | None = None,
    cbar: bool = True,
    cbar_ax: plt.Axes | None = None,
    rasterized: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Model × IE heatmap of party spread (max − min adherence across parties).

    Each cell colour is the **composite** (adherence-index) party spread for a
    model × IE — how unevenly the macro score is distributed across parties. The
    trailing ``*`` flags on the **meso** level instead: a cell is marked when its
    worst rubric-level party spread reaches ``flag_threshold`` (see
    :func:`party_spread_meso`). This mirrors the §1 heatmap — show the macro
    number, flag the rubric-concentrated unevenness the composite average hides.

    Columns are ordered baseline-first, then easiest → hardest (descending mean
    adherence across models), matching the ordering in notebook 02.

    Args:
        df: Tidy scores table (full, or pre-filtered to a single country).
        country: Optional country code; filters ``df`` when given.
        flag_threshold: Meso-flag cutoff; defaults to ``FLAG_THRESHOLD``.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        vmax: Upper colour limit. ``None`` (default) auto-scales to this call's
            own data; pass an explicit value to share one scale across panels
            (e.g. DE vs NL) so cell colours are directly comparable.
        ax: Existing axis; a new figure is created when ``None``.

    Returns:
        The matplotlib ``Axes`` the heatmap was drawn on.
    """
    from src.analysis.aggregate import (
        FLAG_THRESHOLD,
        adherence_index,
        party_spread_meso,
    )

    if flag_threshold is None:
        flag_threshold = FLAG_THRESHOLD
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    df_filtered = df if country is None else df[df["country"] == country]

    spread_df = party_spread_meso(
        df_filtered, by=("model", "country", "ie"), flag_threshold=flag_threshold
    )
    spread_df["model_display"] = spread_df["model"].map(lambda m: names.apply(m, "model"))
    spread_df["ie_display"] = spread_df["ie"].map(lambda ie: names.apply(ie, "ie"))
    spread_df["flag_i"] = spread_df["flag"].astype(int)

    # IE column order: baseline-first, then easiest → hardest (desc mean adherence).
    ie_table = adherence_index(df_filtered, by=["model", "country", "ie"])
    difficulty = ie_table.groupby("ie")["adherence"].mean()
    non_baseline = [ie for ie in difficulty.index if ie != "baseline"]
    ordered_ies_raw = (["baseline"] if "baseline" in difficulty.index else []) + sorted(
        non_baseline, key=lambda ie: difficulty[ie], reverse=True
    )
    ordered_ies_display = [names.apply(ie, "ie") for ie in ordered_ies_raw]

    pivot = spread_df.pivot_table(
        index="model_display", columns="ie_display", values="macro_spread"
    )
    pivot = pivot.reindex(columns=[c for c in ordered_ies_display if c in pivot.columns])

    flag_pivot = spread_df.pivot_table(
        index="model_display", columns="ie_display", values="flag_i"
    ).reindex(columns=pivot.columns)

    # Annotation: plain percentage (no star — flagged cells get a border instead).
    annot = pivot.map(lambda v: f"{round(v * 100)}%" if pd.notna(v) else "")

    if vmax is None:
        vmax = max(float(pivot.values[~pd.isna(pivot.values)].max()), flag_threshold) * 1.05

    if ax is None:
        _, ax = plt.subplots(figsize=(1.15 * len(pivot.columns) + 1.5, 0.7 * len(pivot) + 1.6))
    country_label = f" ({country})" if country else ""
    sns.heatmap(
        pivot,
        annot=annot.values,
        fmt="",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws={"label": "Party spread"},
        annot_kws={"fontsize": ANNOT_FONTSIZE},
        rasterized=rasterized,
        ax=ax,
    )
    # Border on flagged cells (rubric-level spread exceeds threshold).
    _flag_cell_borders(ax, flag_pivot.fillna(0).astype(bool))
    if cbar_ax is not None:
        _style_spread_cbar(ax.collections[0].colorbar)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"Model × IE party spread{country_label}",
        pad=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return ax


def plot_party_spread_publication(
    df: pd.DataFrame,
    *,
    country: str = "DE",
    cell_in: float = 0.52,
    flag_threshold: float | None = None,
    names: DisplayNames | None = None,
    logo_ticks: bool = False,
) -> plt.Figure:
    """Publication wrapper for :func:`plot_party_spread_heatmap` (Fig. H1 / App).

    Full-width figure with square cells guaranteed by absolute-inch placement
    (same technique as :func:`plot_model_ie_robustness`). Works for any country —
    pass ``country="NL"`` for the appendix replication.

    Args:
        df: Full tidy frame (both countries present).
        country: Country code to plot (default ``"DE"``).
        cell_in: Cell side length in inches (default 0.52).
        flag_threshold: Meso-flag cutoff; forwarded to :func:`plot_party_spread_heatmap`.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        logo_ticks: Replace model y-axis labels with logo+name units.

    Returns:
        The Matplotlib Figure at final publication width.
    """
    from src.analysis.aggregate import FLAG_THRESHOLD, adherence_index, party_spread_meso

    if flag_threshold is None:
        flag_threshold = FLAG_THRESHOLD
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES
        names = DISPLAY_NAMES

    # Shared vmax across both countries so the NL appendix figure is comparable.
    # Round up to the nearest 10 pp so the colorbar top/bottom ticks land exactly
    # on a labelled value rather than floating inside the bar.
    spread_all = party_spread_meso(df, flag_threshold=flag_threshold)
    _raw_vmax = max(float(spread_all["macro_spread"].dropna().max()), flag_threshold)
    vmax = math.ceil(_raw_vmax / 0.10) * 0.10

    # Figure out the pivot dimensions so we can size the figure exactly.
    ai = adherence_index(df[df["country"] == country], by=["model", "ie"])
    n_ie = ai["ie"].nunique()
    n_model = ai["model"].nunique()

    # Absolute-inch layout: left margin shrinks when logos replace text labels.
    heat_w = n_ie * cell_in
    heat_h = n_model * cell_in
    left = 0.48 if logo_ticks else 0.80
    bottom, top, cbar_gap, cbar_w, right_pad = 0.60, 0.30, 0.12, 0.14, 0.10
    fig_w = left + heat_w + cbar_gap + cbar_w + right_pad
    fig_h = top + heat_h + bottom

    def _rect(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(_rect(left, bottom, heat_w, heat_h))
    cax = fig.add_axes(_rect(left + heat_w + cbar_gap, bottom, cbar_w, heat_h))

    plot_party_spread_heatmap(
        df, country=country, flag_threshold=flag_threshold, names=names,
        vmax=vmax, cbar=True, cbar_ax=cax, rasterized=True, ax=ax,
    )
    ax.set_title("")

    if logo_ticks:
        from src.analysis.display_names import MODEL_ORDER
        fig.canvas.draw()
        add_model_logo_ticks(ax, MODEL_ORDER, names)

    return fig


def plot_party_spread_pair(
    df: pd.DataFrame,
    *,
    cell_in: float = 0.40,
    flag_threshold: float | None = None,
    names: DisplayNames | None = None,
    logo_ticks: bool = True,
) -> plt.Figure:
    """DE + NL party-spread heatmaps side-by-side with one shared colorbar.

    Compact combined figure for ``figure*`` placement. All font sizes are
    scaled down 10 % relative to the current publication rcParams so the
    denser two-panel layout remains legible. Model labels appear on the DE
    (left) panel only.

    Args:
        df: Full tidy frame (both countries).
        cell_in: Cell side length in inches (default 0.40 ≈ 77 % of the
            single-country default, keeps the combined figure ≤ TEXT_WIDTH_IN).
        flag_threshold: Meso-flag cutoff; forwarded to
            :func:`plot_party_spread_heatmap`. Defaults to ``FLAG_THRESHOLD``.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        logo_ticks: Replace model y-axis labels with logo+name units on the
            DE panel (default ``True``).

    Returns:
        ``Figure`` sized to fit within ``TEXT_WIDTH_IN``.
    """
    from src.analysis.aggregate import FLAG_THRESHOLD, adherence_index, party_spread_meso
    from src.analysis.display_names import DISPLAY_NAMES, MODEL_ORDER

    if flag_threshold is None:
        flag_threshold = FLAG_THRESHOLD
    if names is None:
        names = DISPLAY_NAMES

    spread_all = party_spread_meso(df, flag_threshold=flag_threshold)
    _raw_vmax = max(float(spread_all["macro_spread"].dropna().max()), flag_threshold)
    vmax = math.ceil(_raw_vmax / 0.10) * 0.10

    n_ie = adherence_index(df[df["country"] == "DE"], by=["model", "ie"])["ie"].nunique()
    n_model = adherence_index(df[df["country"] == "DE"], by=["model", "ie"])["model"].nunique()
    heat_w = n_ie * cell_in
    heat_h = n_model * cell_in

    left = 0.48 if logo_ticks else 0.72
    mid_gap = 0.20
    cbar_gap = 0.10
    cbar_w = 0.10   # slimmer than the single-country version
    right_pad = 0.08
    top = 0.28
    bot = 0.55

    fig_w = left + heat_w + mid_gap + heat_w + cbar_gap + cbar_w + right_pad
    fig_h = top + heat_h + bot

    def _r(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Scale all font sizes down 10 % relative to current rcParams.
    _scale = 0.90
    _fs_base    = plt.rcParams.get("font.size", 9)
    _fs_title   = plt.rcParams.get("axes.titlesize", 9)
    _fs_label   = plt.rcParams.get("axes.labelsize", 8)
    _fs_xtick   = plt.rcParams.get("xtick.labelsize", 7.5)
    _fs_ytick   = plt.rcParams.get("ytick.labelsize", 7.5)
    with plt.rc_context({
        "font.size":        _fs_base  * _scale,
        "axes.titlesize":   _fs_title * _scale,
        "axes.labelsize":   _fs_label * _scale,
        "xtick.labelsize":  _fs_xtick * _scale,
        "ytick.labelsize":  _fs_ytick * _scale,
    }):
        ax_de = fig.add_axes(_r(left, bot, heat_w, heat_h))
        ax_nl = fig.add_axes(_r(left + heat_w + mid_gap, bot, heat_w, heat_h))
        cax   = fig.add_axes(_r(left + heat_w + mid_gap + heat_w + cbar_gap, bot, cbar_w, heat_h))

        _common = dict(flag_threshold=flag_threshold, names=names,
                       vmax=vmax, cbar=False, rasterized=True)
        plot_party_spread_heatmap(df, country="DE", ax=ax_de, **_common)
        plot_party_spread_heatmap(df, country="NL", ax=ax_nl, **_common)
        ax_nl.set_yticklabels([])
        # Clear default titles — replaced by inline flag+text below.
        ax_de.set_title("")
        ax_nl.set_title("")

        import matplotlib.colors as _mcolors
        import matplotlib.cm as _cm
        sm = _cm.ScalarMappable(
            cmap="YlOrRd",
            norm=_mcolors.Normalize(vmin=0, vmax=vmax),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        _style_spread_cbar(cb)

    # ── Inline flag + medium-weight country code as panel title ───────────────
    # HPacker places the flag PNG and text side-by-side at the same baseline.
    # FontProperties with fname+weight=500 picks Helvetica Neue Medium from the
    # TTC collection (index 10), which matplotlib cannot register via addfont.
    from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
    import matplotlib.font_manager as _fm
    import numpy as _np
    from PIL import Image as _PILImage

    _flag_dir = Path(__file__).resolve().parents[2] / "analysis" / "assets" / "logos"
    _title_fs = plt.rcParams.get("axes.titlesize", 9) * _scale * 0.88
    _fp_medium = _fm.FontProperties(
        fname="/System/Library/Fonts/HelveticaNeue.ttc",
        weight=500,
        size=_title_fs,
    )
    # zoom so flag cap-height ≈ text cap-height at the target font size
    _zoom = _title_fs / 72.0 * fig.dpi / 160.0 * 0.85

    fig.canvas.draw()   # needed for both HPacker layout and logo ticks

    for _country_code, _ax in (("DE", ax_de), ("NL", ax_nl)):
        _fpath = _flag_dir / f"flag_{_country_code.lower()}.png"
        if _fpath.exists():
            _arr = _np.asarray(_PILImage.open(_fpath).convert("RGBA"))
            _oi = OffsetImage(_arr, zoom=_zoom)
            _oi.image.axes = _ax
            _ta = TextArea(
                _country_code,
                textprops=dict(fontproperties=_fp_medium, color="black"),
            )
            _hpack = HPacker(children=[_oi, _ta], pad=0, sep=3, align="center")
            _ab = AnnotationBbox(
                _hpack,
                xy=(0.5, 1.0),
                xycoords="axes fraction",
                xybox=(0, 6),
                boxcoords="offset points",
                frameon=False,
                box_alignment=(0.5, 0.0),
            )
            _ab.set_clip_on(False)
            _ax.add_artist(_ab)

    if logo_ticks:
        add_model_logo_ticks(ax_de, MODEL_ORDER, names)

    return fig


def plot_ie_sq_heatmap(
    passrate: pd.DataFrame,
    *,
    country: str,
    mode: str = "level",
    names: DisplayNames | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """IE × sub-question pattern heatmap: colour = pass rate, number = model divergence.

    A robustness *diagnostic* (not a fairness signal). Rows are IEs grouped
    baseline-first then by manipulation family (:data:`IE_ROW_GROUPS`, with
    horizontal separators); columns are sub-questions in canonical F→E→I order
    with rubric-block separators. The printed integer is how many models diverge
    from their peers in that cell (the second channel that stops a pooled mean
    hiding one model's collapse). The IE × SQ matrix is sparse by design — cells
    where a sub-question is not active under that IE are masked grey, not zero.

    Read it two ways:
        * **down a column** (one SQ across IEs) — which condition is hardest for
          that sub-question. Only meaningful for SQs active in ≥2 IEs.
        * **across a row** (one IE across SQs) — that condition's failure-mode
          fingerprint. Use ``mode="delta_baseline"`` for this reading so the
          intrinsic per-SQ difficulty is netted out.

    Args:
        passrate: Output of :func:`src.analysis.aggregate.ie_sq_passrate` (must
            carry a ``country`` column; pass it ``by=("country",)``).
        country: Country code to plot (one panel each — parties/coverage differ).
        mode: ``"level"`` (colour = mean pass rate, sequential) or
            ``"delta_baseline"`` (colour = pass rate − that SQ's baseline rate,
            diverging RdBu centred 0; the baseline row is therefore all zeros).
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        vmin: Lower colour bound. ``None`` auto-scales (``level``: data min;
            ``delta_baseline``: ``-vmax``). Pass a shared value across DE/NL.
        vmax: Upper colour bound. ``None`` auto-scales (``level``: 1.0;
            ``delta_baseline``: symmetric max ``|Δ|``). Pass a shared value
            across DE/NL so the two panels are comparable.
        ax: Existing axis; a new figure is created when ``None``.

    Returns:
        The matplotlib ``Axes`` the heatmap was drawn on.

    Raises:
        ValueError: If ``mode`` is unknown, no rows match ``country``, or
            ``delta_baseline`` is requested without a baseline row.

    Usage (notebook 04 — DE + NL side by side, shared scale):
        pr = ie_sq_passrate(df, by=("country",))
        vmin = float(pr["mean_rate"].min())
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        for cc, ax in zip(("DE", "NL"), axes):
            plot_ie_sq_heatmap(pr, country=cc, vmin=vmin, vmax=1.0, ax=ax)
    """
    from src.analysis.display_names import (
        IE_ROW_GROUPS,
        SUBQUESTION_ORDER,
        SUBQUESTION_RUBRIC,
    )

    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    if mode not in {"level", "delta_baseline"}:
        raise ValueError(f"mode must be 'level' or 'delta_baseline', got {mode!r}")

    pr = passrate[passrate["country"] == country]
    if pr.empty:
        raise ValueError(f"No rows in `passrate` for country={country!r}")

    # IE row order from the grouping (present IEs only) + group-boundary rows.
    present_ie = set(pr["ie"])
    ie_order: list[str] = []
    group_boundaries: list[int] = []
    for _, ies in IE_ROW_GROUPS:
        block = [ie for ie in ies if ie in present_ie]
        if not block:
            continue
        if ie_order:
            group_boundaries.append(len(ie_order))
        ie_order.extend(block)
    ie_labels = [names.apply(ie, "ie") for ie in ie_order]

    present_sq = set(pr["subquestion"])
    sq_order = [sq for sq in SUBQUESTION_ORDER if sq in present_sq]
    sq_labels = [names.apply(sq, "subquestion") for sq in sq_order]

    pr = pr.assign(
        ie_label=pr["ie"].map(lambda i: names.apply(i, "ie")),
        sq_label=pr["subquestion"].map(lambda s: names.apply(s, "subquestion")),
    )
    color = pr.pivot_table(
        index="ie_label", columns="sq_label", values="mean_rate"
    ).reindex(index=ie_labels, columns=sq_labels)
    count = pr.pivot_table(
        index="ie_label", columns="sq_label", values="n_diverging"
    ).reindex(index=ie_labels, columns=sq_labels)

    if mode == "delta_baseline":
        baseline_label = names.apply("baseline", "ie")
        if baseline_label not in color.index:
            raise ValueError("delta_baseline mode requires a baseline row in `passrate`")
        color = color.sub(color.loc[baseline_label], axis=1)
        cmap, center = "RdBu", 0.0
        if vmax is None:
            abs_max = color.abs().max().max()
            vmax = max(float(abs_max) if pd.notna(abs_max) else 0.05, 1e-3)
        if vmin is None:
            vmin = -vmax
        cbar_label = "Δ pass rate vs baseline (same SQ)"
    else:
        cmap, center = "viridis", None
        if vmax is None:
            vmax = 1.0
        if vmin is None:
            mn = color.min().min()
            vmin = float(mn) if pd.notna(mn) else 0.0
        cbar_label = "Mean pass rate (all models)"

    # Annotation = divergence count; blank on 0, NaN, or masked (inactive) cells.
    annot = count.copy().astype(object)
    for row in annot.index:
        for col in annot.columns:
            value = count.loc[row, col]
            annot.loc[row, col] = "" if pd.isna(value) or value == 0 else f"{int(value)}"
    annot = annot.where(color.notna(), "")

    if ax is None:
        _, ax = plt.subplots(
            figsize=(0.62 * len(sq_labels) + 3.0, 0.6 * len(ie_labels) + 2.0)
        )
    ax.set_facecolor("0.9")
    sns.heatmap(
        color,
        mask=color.isna(),
        annot=annot.values,
        fmt="",
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        annot_kws={"fontsize": ANNOT_FONTSIZE},
        ax=ax,
    )

    # Rubric-block separators between SQ groups (vertical).
    rubric_seq = [SUBQUESTION_RUBRIC[sq] for sq in sq_order]
    for i in range(1, len(rubric_seq)):
        if rubric_seq[i] != rubric_seq[i - 1]:
            ax.axvline(i, color="#333333", lw=1.8)
    # IE-group separators (horizontal).
    for boundary in group_boundaries:
        ax.axhline(boundary, color="#333333", lw=1.8)

    mode_txt = "Δ vs baseline" if mode == "delta_baseline" else "mean pass rate"
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"IE × sub-question ({country}) — colour = {mode_txt}, "
        "number = # models diverging from peers",
        pad=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return ax


def plot_party_sq_disparity(
    deviation: pd.DataFrame,
    *,
    country: str,
    names: DisplayNames | None = None,
    vmax: float | None = None,
    sq_filter: list[str] | None = None,
    cbar: bool = True,
    cbar_ax: plt.Axes | None = None,
    rasterized: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Party × sub-question disparity heatmap: colour = deviation, number = recurrence.

    Two channels per cell, over a single country. The colour is the party's mean
    leave-one-out deviation across every ``(ie, model)`` context the sub-question
    is active in (red = worse than peers, blue = better); the printed integer is
    how many of those contexts were *flagged*, in the direction matching the
    colour (worse-than-peers count on red cells, better-than-peers count on blue).
    A cell that is strongly coloured and carries a high number — especially one
    that repeats down a party row — is the credible party-specific gap; a one-off
    stays pale with a low/blank count, which is the noise filter.

    Columns run F* -> E* -> I* and split into rubric blocks by vertical rules.
    Feed the output of :func:`src.analysis.aggregate.party_sq_deviation`.

    Args:
        deviation: Per-context deviation frame (``party_sq_deviation`` output).
        country: Country code to plot (parties differ per country, so one call
            each).
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        vmax: Symmetric colour cap (``±vmax``). ``None`` (default) auto-scales to
            this country's own max deviation. Pass a **shared** value (e.g. from
            :func:`src.analysis.aggregate.robust_symmetric_vmax` over both
            countries' ``mean_deviation``) so DE and NL colours are comparable.
        ax: Existing axis; a new figure is created when ``None``.

    Returns:
        The matplotlib ``Axes`` the heatmap was drawn on.
    """
    from src.analysis.aggregate import party_sq_recurrence
    from src.analysis.display_names import (
        PARTY_ORDER,
        SUBQUESTION_ORDER,
        SUBQUESTION_RUBRIC,
    )

    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    dev_c = deviation[deviation["country"] == country]
    rec = party_sq_recurrence(
        dev_c, by=("country", "party", "rubric", "subquestion"), min_recurrence=0
    )

    present_sq = set(rec["subquestion"])
    # sq_filter: restrict to a subset of SQ IDs (e.g. only signal SQs for publication).
    active_sq = {s for s in present_sq if sq_filter is None or s in sq_filter}
    sq_order = [sq for sq in SUBQUESTION_ORDER if sq in active_sq]
    sq_labels = [names.apply(sq, "subquestion") for sq in sq_order]

    present_parties = set(rec["party"])
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    party_labels = [names.apply(p, "party") for p in party_order]

    rec = rec.assign(
        sq_label=rec["subquestion"].map(lambda s: names.apply(s, "subquestion")),
        party_label=rec["party"].map(lambda p: names.apply(p, "party")),
    )
    color = rec.pivot_table(
        index="party_label", columns="sq_label", values="mean_deviation"
    ).reindex(index=party_labels, columns=sq_labels)
    count_neg = rec.pivot_table(
        index="party_label", columns="sq_label", values="recurrence"
    ).reindex(index=party_labels, columns=sq_labels)
    count_pos = rec.pivot_table(
        index="party_label", columns="sq_label", values="recurrence_pos"
    ).reindex(index=party_labels, columns=sq_labels)

    # The annotated count matches the cell's colour direction: a red (negative
    # mean) cell shows worse-than-peers flags, a blue (positive) cell shows
    # better-than-peers flags. Keeps the map symmetric without a second channel.
    annot = color.copy().astype(object)
    for row in annot.index:
        for col in annot.columns:
            mean_dev = color.loc[row, col]
            if pd.isna(mean_dev):
                annot.loc[row, col] = ""
                continue
            value = count_pos.loc[row, col] if mean_dev > 0 else count_neg.loc[row, col]
            annot.loc[row, col] = "" if pd.isna(value) or value == 0 else f"{int(value)}"

    if vmax is None:
        abs_max = color.abs().max().max()
        vmax = max(float(abs_max) if pd.notna(abs_max) else 0.1, 1e-3)

    if ax is None:
        _, ax = plt.subplots(figsize=(0.75 * len(sq_labels) + 2.5, 0.55 * len(party_labels) + 2.0))
    sns.heatmap(
        color,
        annot=annot.values,
        fmt="",
        cmap=DIVERGING_CMAP,
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws={"label": "Mean deviation (red = worse than peers)"},
        rasterized=rasterized,
        ax=ax,
    )

    # Rubric-block separators between SQ groups.
    rubric_seq = [SUBQUESTION_RUBRIC[sq] for sq in sq_order]
    for i in range(1, len(rubric_seq)):
        if rubric_seq[i] != rubric_seq[i - 1]:
            ax.axvline(i, color="#333333", lw=1.8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"Party × sub-question disparity ({country})  "
        "— colour = mean LOO deviation, number = flagged (IE, model) contexts "
        "(direction matches colour)",
        pad=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return ax


# 6 SQs with max |party deviation| > 0.03 in DE — confirmed from data in notebook 03.
SIGNAL_SUBQUESTIONS: list[str] = [
    "false_synthesis",
    "context_transparency",
    "epistemic_hedging",
    "noise_contamination",
    "sanitization",
    "position_representation",
]


def plot_party_sq_disparity_publication(
    deviation: pd.DataFrame,
    *,
    country: str = "DE",
    sq_filter: list[str] | None = None,
    names: DisplayNames | None = None,
    logo_ticks: bool = False,
) -> plt.Figure:
    """Publication wrapper for :func:`plot_party_sq_disparity` (Fig. H2).

    Full-width (``TEXT_WIDTH_IN``) single-panel figure. Pruned to the 6 SQs
    with signal (``SIGNAL_SUBQUESTIONS``) by default; override via ``sq_filter``.
    Uses ``DIVERGING_CMAP`` (``RdBu_r``), strips the auto-title, inherits rcParams
    font sizes, rasterized mesh, colourbar pinned to cell block height.

    Args:
        deviation: Per-context deviation frame (``party_sq_deviation`` output).
        country: Country code to plot.
        sq_filter: SQ IDs to keep; defaults to :data:`SIGNAL_SUBQUESTIONS`.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        logo_ticks: Replace party y-axis labels with logo icons.

    Returns:
        The Matplotlib Figure at final publication width.
    """
    if sq_filter is None:
        sq_filter = SIGNAL_SUBQUESTIONS
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES
        names = DISPLAY_NAMES

    from src.analysis.aggregate import party_sq_recurrence
    from src.analysis.display_names import PARTY_ORDER

    dev_c = deviation[deviation["country"] == country]
    rec = party_sq_recurrence(
        dev_c, by=("country", "party", "rubric", "subquestion"), min_recurrence=0
    )

    # Only party count needed here — sq ordering is delegated to plot_party_sq_disparity.
    present_parties = set(rec["party"])
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    n_party = len(party_order)

    # 0.32in per row keeps it rectangular rather than block-like.
    fig_h = 0.32 * n_party + 1.4
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, fig_h))

    plot_party_sq_disparity(
        deviation, country=country, names=names, sq_filter=sq_filter,
        cbar=False, rasterized=True, ax=ax,
    )
    ax.set_title("")

    # Manual colourbar pinned to the exact cell-block height.
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes((pos.x1 + 0.01, pos.y0, 0.012, pos.height))
    from src.analysis.aggregate import robust_symmetric_vmax
    active_devs = dev_c.loc[dev_c["subquestion"].isin(sq_filter), "deviation"]
    # Round up to nearest 5 pp so the diverging bar ends exactly on a tick.
    vmax = math.ceil(robust_symmetric_vmax(active_devs) / 0.05) * 0.05
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(
        cmap=DIVERGING_CMAP,
        norm=mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Mean deviation")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(direction="in", length=2, width=0.5)

    if logo_ticks:
        add_party_logo_ticks(ax, party_order, names)

    return fig


def plot_party_sq_breakdown(
    deviation: pd.DataFrame,
    *,
    country: str,
    subquestion: str,
    names: DisplayNames | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Per-IE × model breakdown of one sub-question's party deviations.

    The drill behind a flagged disparity cell: one model panel each, rows =
    party, columns = IE, colour = leave-one-out deviation, ``*`` on flagged
    contexts. Shows where a recurring gap actually comes from without collapsing
    the IE or model dimension.

    Args:
        deviation: Per-context deviation frame (``party_sq_deviation`` output).
        country: Country code to plot.
        subquestion: Raw sub-question id to drill into.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        vmax: Symmetric colour cap (``±vmax``). ``None`` (default) auto-scales to
            this slice's own max deviation. Pass a **shared** value (e.g. from
            :func:`src.analysis.aggregate.robust_symmetric_vmax` over both
            countries' per-context ``deviation``) so drilldowns are comparable.

    Returns:
        The matplotlib ``Figure`` holding one heatmap per model.

    Raises:
        ValueError: If no rows match ``country`` and ``subquestion``.
    """
    from src.analysis.display_names import PARTY_ORDER

    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    sub = deviation[
        (deviation["country"] == country) & (deviation["subquestion"] == subquestion)
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows for country={country!r}, subquestion={subquestion!r}")

    ie_canonical = [
        "baseline",
        "noise",
        "prior_conflict",
        "clarity",
        "availability",
        "consistency",
    ]
    models = sorted(sub["model"].unique(), key=lambda m: names.apply(m, "model"))
    present_parties = set(sub["party"])
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    party_labels = [names.apply(p, "party") for p in party_order]
    present_ie = set(sub["ie"])
    ie_order = [ie for ie in ie_canonical if ie in present_ie]
    ie_labels = [names.apply(ie, "ie") for ie in ie_order]

    sub["party_label"] = sub["party"].map(lambda p: names.apply(p, "party"))
    sub["ie_label"] = sub["ie"].map(lambda i: names.apply(i, "ie"))
    sub["flag_i"] = sub["flag"].astype(int)

    if vmax is None:
        abs_max = sub["deviation"].abs().max()
        vmax = max(float(abs_max) if pd.notna(abs_max) else 0.1, 1e-3)

    n = len(models)
    fig, axes = plt.subplots(
        1,
        n,
        sharey=True,
        squeeze=False,
        figsize=(max(1.0 * len(ie_labels) + 1.5, 3.0) * n, 0.5 * len(party_labels) + 2.0),
    )
    for col, model in enumerate(models):
        ax = axes[0][col]
        m = sub[sub["model"] == model]
        dev_p = m.pivot_table(index="party_label", columns="ie_label", values="deviation").reindex(
            index=party_labels, columns=ie_labels
        )
        flag_p = m.pivot_table(index="party_label", columns="ie_label", values="flag_i").reindex(
            index=party_labels, columns=ie_labels
        )
        annot = dev_p.copy().astype(object)
        for row in annot.index:
            for c in annot.columns:
                value = dev_p.loc[row, c]
                if pd.isna(value):
                    annot.loc[row, c] = ""
                    continue
                star = "*" if flag_p.loc[row, c] == 1 else ""
                annot.loc[row, c] = f"{value:+.2f}{star}"
        is_last = col == n - 1
        sns.heatmap(
            dev_p,
            annot=annot.values,
            fmt="",
            cmap="RdBu",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar=is_last,
            cbar_kws={"label": "LOO deviation"} if is_last else None,
            ax=ax,
        )
        ax.set_title(names.apply(model, "model"))
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Append the cell's all-party mean pass rate per IE — context for whether
        # a flat/positive column is just a ceiling (μ ~ 1.00) the deviation rides on.
        ie_mean = m.groupby("ie_label", observed=True)["rate"].mean()
        ax.set_xticklabels(
            [f"{ie}\nμ={ie_mean.get(ie, float('nan')):.2f}" for ie in ie_labels],
            rotation=30,
            ha="right",
        )

    sq_label = names.apply(subquestion, "subquestion")
    fig.suptitle(
        f"{sq_label} — per-IE × model party deviation ({country})  (* = flagged)",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_cell_breakdown(
    df: pd.DataFrame,
    *,
    model: str,
    ie: str,
    country: str,
    level: str = "rubric",
    names: DisplayNames | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Party × rubric (or sub-question) deviation inside one model × IE cell.

    The middle-zoom drill behind a §2 spread flag: fix one ``(model, ie)`` cell
    and show each party's leave-one-out deviation from the other parties,
    decomposed by rubric (``level="rubric"``, default) or sub-question
    (``level="subquestion"``). Colour = deviation (red = this party below its
    peers in this cell); annotation = the party's actual pass rate, with ``*`` on
    flagged cells. IE-specific by construction — the cross-IE view is §3, where a
    one-off cell effect averages out.

    Args:
        df: Tidy scores table.
        model: Raw model id to inspect.
        ie: Raw IE id to inspect.
        country: Country code to inspect.
        level: ``"rubric"`` (default) or ``"subquestion"`` granularity.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        vmax: Symmetric colour limit (±vmax); auto from data when ``None``. Pass an
            explicit value to make side-by-side panels comparable.
        ax: Existing axis; a new figure is created when ``None``.

    Returns:
        The matplotlib ``Axes`` the heatmap was drawn on.

    Raises:
        ValueError: If ``level`` is unknown or no rows match the cell.
    """
    from src.analysis.aggregate import party_sq_deviation
    from src.analysis.display_names import (
        PARTY_ORDER,
        SUBQUESTION_ORDER,
        SUBQUESTION_RUBRIC,
    )

    if level not in ("rubric", "subquestion"):
        raise ValueError(f"level must be 'rubric' or 'subquestion', got {level!r}")
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES

        names = DISPLAY_NAMES

    cell = df[(df["country"] == country) & (df["model"] == model) & (df["ie"] == ie)]
    if cell.empty:
        raise ValueError(f"No rows for country={country!r}, model={model!r}, ie={ie!r}")

    if level == "rubric":
        dev = party_sq_deviation(cell, by=("country", "model", "ie", "rubric"))
        col_field = "rubric"
        col_raw_order = ["faithfulness", "impartiality", "epistemic_calibration"]
        present_cols = [c for c in col_raw_order if c in set(dev["rubric"])]
    else:
        dev = party_sq_deviation(cell)
        col_field = "subquestion"
        present_cols = [s for s in SUBQUESTION_ORDER if s in set(dev["subquestion"])]
    col_labels = [names.apply(c, col_field) for c in present_cols]

    present_parties = set(dev["party"])
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    party_labels = [names.apply(p, "party") for p in party_order]

    dev = dev.assign(
        col_label=dev[col_field].map(lambda v: names.apply(v, col_field)),
        party_label=dev["party"].map(lambda p: names.apply(p, "party")),
        # Star either direction; the colour already shows worse (red) vs better (blue).
        flag_i=(dev["flag_neg"] | dev["flag_pos"]).astype(int),
    )
    color = dev.pivot_table(index="party_label", columns="col_label", values="deviation").reindex(
        index=party_labels, columns=col_labels
    )
    rate = dev.pivot_table(index="party_label", columns="col_label", values="rate").reindex(
        index=party_labels, columns=col_labels
    )
    flag = dev.pivot_table(index="party_label", columns="col_label", values="flag_i").reindex(
        index=party_labels, columns=col_labels
    )

    annot = rate.copy().astype(object)
    for row in annot.index:
        for c in annot.columns:
            value = rate.loc[row, c]
            if pd.isna(value):
                annot.loc[row, c] = ""
                continue
            star = "*" if flag.loc[row, c] == 1 else ""
            annot.loc[row, c] = f"{value:.2f}{star}"

    if vmax is None:
        abs_max = color.abs().max().max()
        vmax = max(float(abs_max) if pd.notna(abs_max) else 0.1, 1e-3)

    if ax is None:
        _, ax = plt.subplots(figsize=(0.95 * len(col_labels) + 2.5, 0.5 * len(party_labels) + 1.8))
    sns.heatmap(
        color,
        annot=annot.values,
        fmt="",
        cmap="RdBu",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "LOO deviation (red = below peers)"},
        ax=ax,
    )
    if level == "subquestion":
        rubric_seq = [SUBQUESTION_RUBRIC[s] for s in present_cols]
        for i in range(1, len(rubric_seq)):
            if rubric_seq[i] != rubric_seq[i - 1]:
                ax.axvline(i, color="#333333", lw=1.8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{names.apply(model, 'model')} × {names.apply(ie, 'ie')} ({country}) — party × {level}",
        pad=8,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ax


def _build_delta_pivot(
    delta_df: pd.DataFrame,
    *,
    country: str,
    names: DisplayNames,
    party_order: list[str],
    sq_order: list[str],
) -> pd.DataFrame:
    """Build the party x sub-question delta_rate pivot for a country slice.

    Args:
        delta_df: Full multi-country delta frame (``paired_delta`` output).
        country: Country code to slice.
        names: Display-name registry.
        party_order: Raw party IDs in canonical row order.
        sq_order: Raw sub-question IDs in canonical column order.

    Returns:
        DataFrame indexed by party display labels, columns by SQ display labels.
    """
    df_c = delta_df[delta_df["country"] == country].copy()
    df_c["_party_label"] = df_c["party"].map(lambda p: names.apply(p, "party"))
    df_c["_sq_label"] = df_c["subquestion"].map(lambda s: names.apply(s, "subquestion"))
    party_labels = [names.apply(p, "party") for p in party_order]
    sq_labels = [names.apply(sq, "subquestion") for sq in sq_order]
    return df_c.pivot_table(index="_party_label", columns="_sq_label", values="delta_rate").reindex(
        index=party_labels, columns=sq_labels
    )


def _build_delta_annot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Build annotation array for a delta pivot: ±pp integers, blank if |Δ|<1pp.

    Args:
        pivot: Party x SQ float pivot of delta_rate values (0–1 scale).

    Returns:
        Object-dtype DataFrame of the same shape with annotation strings.
    """
    annot = pivot.copy().astype(object)
    for row in pivot.index:
        for col in pivot.columns:
            val = pivot.loc[row, col]
            if pd.isna(val) or abs(float(val) * 100) < 1.0:
                annot.loc[row, col] = ""
            else:
                annot.loc[row, col] = f"{float(val) * 100:+.0f}"
    return annot


def _draw_delta_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    sq_order: list[str],
    *,
    vmax: float,
    cbar: bool = True,
) -> None:
    """Render a house-style RdBu diverging heatmap onto an existing axis.

    Uses the canonical house style: ``cmap="RdBu"``, ``center=0.0``,
    ``vmin=-vmax/vmax=+vmax``, ``linewidths=0.5``, rubric-block separators.
    Red = negative/worse, blue = positive/better.

    Args:
        ax: Target matplotlib axis.
        pivot: Party x SQ delta_rate pivot (display-labelled rows/columns).
        sq_order: Raw sub-question IDs corresponding to pivot columns (used to
            draw rubric separators via ``SUBQUESTION_RUBRIC``).
        vmax: Symmetric colour cap; must be positive.
        cbar: Draw the colorbar when ``True`` (default).
    """
    from src.analysis.display_names import SUBQUESTION_RUBRIC

    safe_vmax = vmax if vmax > 1e-9 else 1e-6
    annot = _build_delta_annot(pivot)
    sns.heatmap(
        pivot,
        annot=annot.values,
        fmt="",
        cmap="RdBu",
        center=0.0,
        vmin=-safe_vmax,
        vmax=safe_vmax,
        linewidths=0.5,
        linecolor="white",
        cbar=cbar,
        cbar_kws={"label": r"$\Delta$ pass rate (pp)", "shrink": 0.85} if cbar else None,
        ax=ax,
    )
    # Rubric-block vertical separators.
    rubric_seq = [SUBQUESTION_RUBRIC[sq] for sq in sq_order]
    for i in range(1, len(rubric_seq)):
        if rubric_seq[i] != rubric_seq[i - 1]:
            ax.axvline(i, color="#333333", lw=1.8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")


def plot_ablation_delta_heatmap(
    delta_df: pd.DataFrame,
    *,
    country: str,
    contrast_label: str,
    names: DisplayNames = None,  # type: ignore[assignment]
) -> plt.Figure:
    """Party x sub-question delta-rate heatmap for a single country.

    Renders a single diverging heatmap of the signed pass-rate change
    (``delta_rate``) for every party x sub-question combination in *country*.
    The colour scale is computed over the **entire** ``delta_df`` (all countries)
    before filtering, so figures produced for different countries on the same
    contrast share one comparable scale.

    Red = negative/worse, blue = positive/better (``cmap="RdBu"``, house style).
    Rubric-block separators are drawn between the faithfulness,
    epistemic-calibration, and impartiality groups, matching the convention in
    :func:`plot_party_sq_disparity`.

    Args:
        delta_df: Tidy frame with columns ``country`` (``"DE"``/``"NL"``),
            ``party`` (raw id), ``subquestion`` (raw id), and ``delta_rate``
            (signed float; already aggregated over models).  The full
            multi-country frame must be passed — filtering happens inside.
        country: Country code to plot (e.g. ``"DE"`` or ``"NL"``).
        contrast_label: Axes title prefix, e.g.
            ``r"$\\Delta$ pass rate: anon $-$ full (Baseline)"``.
            The function appends ``" ({country})"`` automatically.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.

    Returns:
        The matplotlib ``Figure`` containing a single heatmap ``Axes`` and
        its own colorbar.

    Raises:
        ValueError: If ``delta_df`` contains no rows for ``country``.

    Usage:
        fig = plot_ablation_delta_heatmap(
            delta_df,
            country="DE",
            contrast_label=r"$\\Delta$ pass rate: anon $-$ full (Baseline)",
        )
        fig.savefig("ablation_delta_de.png", bbox_inches="tight")
    """
    from src.analysis.aggregate import robust_symmetric_vmax
    from src.analysis.display_names import (
        DISPLAY_NAMES as _DISPLAY_NAMES,
    )
    from src.analysis.display_names import (
        PARTY_ORDER,
        SUBQUESTION_ORDER,
    )

    if names is None:
        names = _DISPLAY_NAMES

    # ── Shared colour scale (computed over ALL countries before filtering) ────
    vmax = robust_symmetric_vmax(delta_df["delta_rate"])

    # ── Filter to the requested country ──────────────────────────────────────
    df_c = delta_df[delta_df["country"] == country]
    if df_c.empty:
        raise ValueError(f"No rows found for country={country!r} in delta_df.")

    # ── Ordered party / SQ lists for this country ─────────────────────────────
    present_parties = set(df_c["party"].unique())
    party_order = [p for p in PARTY_ORDER if p in present_parties]

    present_sq = set(df_c["subquestion"].unique())
    sq_order = [sq for sq in SUBQUESTION_ORDER if sq in present_sq]

    pivot = _build_delta_pivot(
        delta_df,
        country=country,
        names=names,
        party_order=party_order,
        sq_order=sq_order,
    )

    # ── Figure sizing: width scales with SQ count, height with party count ────
    n_sq = len(sq_order)
    n_parties = len(party_order)
    fig_w = 0.75 * n_sq + 2.5
    fig_h = 0.55 * n_parties + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    _draw_delta_heatmap(ax, pivot, sq_order, vmax=vmax, cbar=True)
    ax.set_title(f"{contrast_label} ({country})", pad=8)

    fig.tight_layout()
    return fig


def plot_ablation_delta_with_actuals(
    delta_df: pd.DataFrame,
    *,
    country: str,
    contrast_label: str,
    names: DisplayNames = None,  # type: ignore[assignment]
) -> plt.Figure:
    """Paired 1x2 figure: actual full-condition pass rate (left) + delta (right).

    Both panels share the same party rows and sub-question columns, with rubric
    separators on both. The left panel shows the absolute pass rate under the
    full condition (``rate_full``), using a sequential colormap on a 0-1 scale.
    The right panel shows the signed delta (``delta_rate``) using the house-style
    RdBu diverging palette (red = negative/worse, blue = positive/better).

    The colour scale for the delta panel is computed over the **entire**
    ``delta_df`` (all countries) before filtering, so DE and NL are comparable.

    Args:
        delta_df: Output of ``paired_delta(..., by=("country","party","subquestion"))``.
            Must contain columns ``country``, ``party``, ``subquestion``,
            ``rate_full``, and ``delta_rate``.  The full multi-country frame must
            be passed; filtering to ``country`` happens inside.
        country: Country code to plot (e.g. ``"DE"`` or ``"NL"``).
        contrast_label: Right-panel title prefix; caller should use raw TeX
            strings, e.g. ``r"$\\Delta$ anon $-$ full (Baseline)"``.
            The function appends ``" ({country})"`` automatically.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.

    Returns:
        The matplotlib ``Figure`` containing two heatmap panels.

    Raises:
        ValueError: If the country slice of ``delta_df`` is empty.
    """
    from src.analysis.aggregate import robust_symmetric_vmax
    from src.analysis.display_names import (
        DISPLAY_NAMES as _DISPLAY_NAMES,
    )
    from src.analysis.display_names import (
        PARTY_ORDER,
        SUBQUESTION_ORDER,
    )

    if names is None:
        names = _DISPLAY_NAMES

    # ── Filter and validate ───────────────────────────────────────────────────
    df_c = delta_df[delta_df["country"] == country].copy()
    if df_c.empty:
        raise ValueError(f"No rows found for country={country!r} in delta_df.")

    # ── Shared axes ordering derived ONCE from the country slice ──────────────
    present_parties = set(df_c["party"].unique())
    party_order = [p for p in PARTY_ORDER if p in present_parties]

    present_sq = set(df_c["subquestion"].unique())
    sq_order = [sq for sq in SUBQUESTION_ORDER if sq in present_sq]

    party_labels = [names.apply(p, "party") for p in party_order]
    sq_labels = [names.apply(sq, "subquestion") for sq in sq_order]

    df_c["_party_label"] = df_c["party"].map(lambda p: names.apply(p, "party"))
    df_c["_sq_label"] = df_c["subquestion"].map(lambda s: names.apply(s, "subquestion"))

    # ── Left panel: actual pass rate under full condition ─────────────────────
    rate_pivot = df_c.pivot_table(
        index="_party_label", columns="_sq_label", values="rate_full"
    ).reindex(index=party_labels, columns=sq_labels)

    rate_annot = rate_pivot.copy().astype(object)
    for row in rate_pivot.index:
        for col in rate_pivot.columns:
            val = rate_pivot.loc[row, col]
            rate_annot.loc[row, col] = "" if pd.isna(val) else f"{int(round(float(val) * 100))}"

    # ── Right panel: delta (all-country vmax for comparability) ──────────────
    delta_vmax = robust_symmetric_vmax(delta_df["delta_rate"])
    delta_pivot = _build_delta_pivot(
        delta_df,
        country=country,
        names=names,
        party_order=party_order,
        sq_order=sq_order,
    )

    # ── Figure: 1x2, left wider for colorbar differences ─────────────────────
    from src.analysis.display_names import SUBQUESTION_RUBRIC

    n_sq = len(sq_order)
    n_parties = len(party_order)
    panel_w = 0.75 * n_sq + 2.5
    panel_h = 0.55 * n_parties + 2.0
    fig, (ax_rate, ax_delta) = plt.subplots(1, 2, figsize=(panel_w * 2 + 1.0, panel_h), sharey=True)

    # Left: sequential viridis 0-1, annotate as integer %
    sns.heatmap(
        rate_pivot,
        annot=rate_annot.values,
        fmt="",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pass rate (\\%)", "shrink": 0.85},
        ax=ax_rate,
    )
    # Rubric separators on left panel too.
    rubric_seq = [SUBQUESTION_RUBRIC[sq] for sq in sq_order]
    for i in range(1, len(rubric_seq)):
        if rubric_seq[i] != rubric_seq[i - 1]:
            ax_rate.axvline(i, color="#333333", lw=1.8)
    ax_rate.set_xlabel("")
    ax_rate.set_ylabel("")
    # TeX-safe: use -- (en-dash) not em-dash, no unicode.
    ax_rate.set_title(f"Actual pass rate -- full ({country})", pad=8)
    plt.setp(ax_rate.get_xticklabels(), rotation=35, ha="right")

    # Right: delta heatmap via shared helper.
    _draw_delta_heatmap(ax_delta, delta_pivot, sq_order, vmax=delta_vmax, cbar=True)
    ax_delta.set_title(f"{contrast_label} ({country})", pad=8)

    fig.tight_layout()
    return fig


def plot_ablation_sq_model_delta(
    delta_model_df: pd.DataFrame,
    *,
    country: str,
    subquestion: str,
    contrast_label: str,
    names: DisplayNames = None,  # type: ignore[assignment]
) -> plt.Figure:
    """Model x party delta heatmap for a single sub-question, with a Mean column.

    Renders one heatmap where rows = party (``PARTY_ORDER`` for the country),
    columns = individual models (ordered by display name) plus a final ``Mean``
    column = mean ``delta_rate`` across models per party.  A thick separator
    rule divides the ``Mean`` column from the model columns.

    House-style RdBu colouring (red = negative/worse, blue = positive/better).
    The colour scale is computed over models + Mean together so the Mean column
    does not clip.

    Args:
        delta_model_df: Output of ``paired_delta(..., by=("country","model",
            "party","subquestion"))``.  Must contain columns ``country``,
            ``model``, ``party``, ``subquestion``, and ``delta_rate``.
        country: Country code to plot (e.g. ``"DE"`` or ``"NL"``).
        subquestion: Raw sub-question id to drill into (e.g.
            ``"sanitization"``).
        contrast_label: Axes title infix; caller should use raw TeX strings,
            e.g. ``r"$\\Delta$ anon $-$ full (Baseline)"``.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.

    Returns:
        The matplotlib ``Figure`` containing a single heatmap.

    Raises:
        ValueError: If the (country, subquestion) slice is empty.
    """
    from src.analysis.aggregate import robust_symmetric_vmax
    from src.analysis.display_names import (
        DISPLAY_NAMES as _DISPLAY_NAMES,
    )
    from src.analysis.display_names import PARTY_ORDER

    if names is None:
        names = _DISPLAY_NAMES

    # ── Filter to the (country, subquestion) slice ────────────────────────────
    df_s = delta_model_df[
        (delta_model_df["country"] == country) & (delta_model_df["subquestion"] == subquestion)
    ].copy()
    if df_s.empty:
        raise ValueError(f"No rows for country={country!r}, subquestion={subquestion!r}.")

    # ── Ordered rows and columns ──────────────────────────────────────────────
    present_parties = set(df_s["party"].unique())
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    party_labels = [names.apply(p, "party") for p in party_order]

    # Models ordered by display name for determinism.
    models_raw = sorted(df_s["model"].unique(), key=lambda m: names.apply(m, "model"))
    model_labels = [names.apply(m, "model") for m in models_raw]

    df_s["_party_label"] = df_s["party"].map(lambda p: names.apply(p, "party"))
    df_s["_model_label"] = df_s["model"].map(lambda m: names.apply(m, "model"))

    pivot = df_s.pivot_table(
        index="_party_label", columns="_model_label", values="delta_rate"
    ).reindex(index=party_labels, columns=model_labels)

    # Append Mean column (mean across all models per party).
    pivot["Mean"] = pivot.mean(axis=1)
    all_cols = model_labels + ["Mean"]

    # vmax includes Mean so no clipping.
    flat_vals = pivot[all_cols].values.ravel()
    non_nan = flat_vals[~pd.isnull(flat_vals)]
    vmax = robust_symmetric_vmax(pd.Series(non_nan)) if len(non_nan) else 0.1

    # Annotation: ±pp, suppress |Δ|<1pp.
    annot = pivot[all_cols].copy().astype(object)
    for row in pivot.index:
        for col in all_cols:
            val = pivot.loc[row, col]
            if pd.isna(val) or abs(float(val) * 100) < 1.0:
                annot.loc[row, col] = ""
            else:
                annot.loc[row, col] = f"{float(val) * 100:+.0f}"

    # ── Figure ────────────────────────────────────────────────────────────────
    n_cols = len(all_cols)
    n_parties = len(party_order)
    fig_w = 0.75 * n_cols + 2.5
    fig_h = 0.55 * n_parties + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    safe_vmax = vmax if vmax > 1e-9 else 1e-6
    sns.heatmap(
        pivot[all_cols],
        annot=annot.values,
        fmt="",
        cmap="RdBu",
        center=0.0,
        vmin=-safe_vmax,
        vmax=safe_vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": r"$\Delta$ pass rate (pp)", "shrink": 0.85},
        ax=ax,
    )

    # Separator before Mean column (visual vocabulary: same style as rubric lines).
    ax.axvline(len(model_labels), color="#333333", lw=1.8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    # TeX-safe title: use x not unicode x, -- not em-dash.
    sq_label = names.apply(subquestion, "subquestion")
    ax.set_title(f"{sq_label} -- Model x Party {contrast_label} ({country})", pad=8)

    fig.tight_layout()
    return fig


def plot_ablation_sq_model_delta_paired(
    delta_model_df: pd.DataFrame,
    *,
    subquestion: str,
    contrast_label: str,
    countries: list[str] = None,  # type: ignore[assignment]
    names: DisplayNames = None,  # type: ignore[assignment]
) -> plt.Figure:
    """Side-by-side Model x Party delta heatmaps for DE and NL on a shared colour scale.

    Same layout as :func:`plot_ablation_sq_model_delta` but renders both countries
    in a single 1×2 figure with a shared ``vmax`` so cell colours are directly
    comparable.

    Usage (notebook)::

        fig = plot_ablation_sq_model_delta_paired(
            delta_model_anon_baseline,
            subquestion="sanitization",
            contrast_label=r"$\\Delta$ anon $-$ full (Baseline)",
        )
        save_plot(fig=fig, ..., plot_name="ablation_sq_model_sanitization_anon")

    Args:
        delta_model_df: Output of ``paired_delta(..., by=("country","model",
            "party","subquestion"))``.
        subquestion: Raw sub-question id (e.g. ``"sanitization"``).
        contrast_label: Title infix; use raw TeX strings.
        countries: Country codes to show, in order. Defaults to ``["DE", "NL"]``.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.

    Returns:
        A ``Figure`` with two side-by-side heatmap axes (shared colour scale,
        single shared colour bar on the right).

    Raises:
        ValueError: If the subquestion slice is empty for all requested countries.
    """
    from src.analysis.aggregate import robust_symmetric_vmax
    from src.analysis.display_names import DISPLAY_NAMES as _DISPLAY_NAMES
    from src.analysis.display_names import PARTY_ORDER

    if names is None:
        names = _DISPLAY_NAMES
    if countries is None:
        countries = ["DE", "NL"]

    # ── Build per-country pivots and compute shared vmax ──────────────────────
    pivots: dict[str, pd.DataFrame] = {}
    all_cols_per: dict[str, list[str]] = {}

    for country in countries:
        df_s = delta_model_df[
            (delta_model_df["country"] == country) & (delta_model_df["subquestion"] == subquestion)
        ].copy()
        if df_s.empty:
            raise ValueError(f"No rows for country={country!r}, subquestion={subquestion!r}.")

        present_parties = set(df_s["party"].unique())
        party_order = [p for p in PARTY_ORDER if p in present_parties]
        party_labels = [names.apply(p, "party") for p in party_order]

        models_raw = sorted(df_s["model"].unique(), key=lambda m: names.apply(m, "model"))
        model_labels = [names.apply(m, "model") for m in models_raw]

        df_s["_party_label"] = df_s["party"].map(lambda p: names.apply(p, "party"))
        df_s["_model_label"] = df_s["model"].map(lambda m: names.apply(m, "model"))

        pivot = df_s.pivot_table(
            index="_party_label", columns="_model_label", values="delta_rate"
        ).reindex(index=party_labels, columns=model_labels)
        pivot["Mean"] = pivot.mean(axis=1)
        all_cols = model_labels + ["Mean"]

        pivots[country] = pivot
        all_cols_per[country] = all_cols

    # Shared vmax across both countries.
    all_vals = []
    for country in countries:
        all_vals.extend(pivots[country][all_cols_per[country]].values.ravel().tolist())
    non_nan = [v for v in all_vals if not pd.isnull(v)]
    vmax = robust_symmetric_vmax(pd.Series(non_nan)) if non_nan else 0.1
    safe_vmax = vmax if vmax > 1e-9 else 1e-6

    sq_label = names.apply(subquestion, "subquestion")

    # ── Figure: 1 row × n_countries columns, shared cbar on far right ─────────
    max_parties = max(len(pivots[c].index) for c in countries)
    max_cols = max(len(all_cols_per[c]) for c in countries)
    fig_w = (0.75 * max_cols + 2.2) * len(countries) + 0.8  # +0.8 for shared cbar
    fig_h = 0.55 * max_parties + 2.0

    fig, axes = plt.subplots(1, len(countries), figsize=(fig_w, fig_h))
    if len(countries) == 1:
        axes = [axes]

    for ax, country in zip(axes, countries, strict=False):
        pivot = pivots[country]
        all_cols = all_cols_per[country]
        is_last = country == countries[-1]

        annot = pivot[all_cols].copy().astype(object)
        for row in pivot.index:
            for col in all_cols:
                val = pivot.loc[row, col]
                if pd.isna(val) or abs(float(val) * 100) < 1.0:
                    annot.loc[row, col] = ""
                else:
                    annot.loc[row, col] = f"{float(val) * 100:+.0f}"

        sns.heatmap(
            pivot[all_cols],
            annot=annot.values,
            fmt="",
            cmap="RdBu",
            center=0.0,
            vmin=-safe_vmax,
            vmax=safe_vmax,
            linewidths=0.5,
            linecolor="white",
            cbar=is_last,
            cbar_kws={"label": r"$\Delta$ pass rate (pp)", "shrink": 0.85} if is_last else None,
            ax=ax,
        )

        # Separator before Mean column.
        ax.axvline(len(all_cols) - 1, color="#333333", lw=1.8)

        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_title(f"{sq_label} -- {country} -- {contrast_label}", pad=8)

    fig.tight_layout()
    return fig


# ── Canonical condition ordering for slope charts ─────────────────────────────
_CONDITION_ORDER: list[str] = ["full", "anon", "english"]


def plot_ablation_slope(
    slope_df: pd.DataFrame,
    *,
    country: str,
    subquestion_label: str,
    names: DisplayNames = None,  # type: ignore[assignment]
) -> plt.Figure:
    """Party pass-rate slope chart across ablation conditions for a single country.

    Shows one line per party (coloured by ``party_palette``) with bootstrap-CI
    error bars, x-axis = condition (canonical order), y-axis = pass rate 0-1.05.
    A small per-party horizontal jitter prevents overlapping error bars.

    The legend sits **outside the axes on the right** so it never overlaps
    the lines, and lists only the parties actually present in *country*.

    Args:
        slope_df: Tidy frame with columns ``country``, ``party`` (raw id),
            ``condition`` (e.g. ``"full"``, ``"anon"``, ``"english"``), ``rate``
            (0-1), ``ci_low``, ``ci_high``.  The full multi-country frame may be
            passed; filtering to *country* happens inside.
        country: Country code to plot (e.g. ``"DE"`` or ``"NL"``).
        subquestion_label: Axes title prefix, e.g.
            ``r"Sanitization (I4)"``.
            The function appends ``" ({country})"`` automatically.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.

    Returns:
        The matplotlib ``Figure`` containing a single slope ``Axes`` with an
        outside-right legend.

    Raises:
        ValueError: If ``slope_df`` contains no rows for ``country``.

    Usage:
        fig = plot_ablation_slope(
            slope_df,
            country="DE",
            subquestion_label=r"Sanitization (I4)",
        )
        fig.savefig("ablation_slope_de.png", bbox_inches="tight")
    """
    from src.analysis.display_names import (
        DISPLAY_NAMES as _DISPLAY_NAMES,
    )
    from src.analysis.display_names import (
        PARTY_ORDER,
        party_palette,
    )

    if names is None:
        names = _DISPLAY_NAMES

    # ── Filter to the requested country ──────────────────────────────────────
    df_c = slope_df[slope_df["country"] == country].copy()
    if df_c.empty:
        raise ValueError(f"No rows found for country={country!r} in slope_df.")

    # ── Conditions present in data, in canonical order ────────────────────────
    present_conditions = set(df_c["condition"].unique())
    conditions = [c for c in _CONDITION_ORDER if c in present_conditions]
    cond_to_x: dict[str, int] = {c: i for i, c in enumerate(conditions)}

    palette_map = party_palette(names)

    # ── Figure: single panel, right margin reserved for the outside legend ────
    fig, ax = plt.subplots(figsize=(6, 4))

    present_parties = set(df_c["party"].unique())
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    n_parties = len(party_order)

    for idx, party_raw in enumerate(party_order):
        label = names.apply(party_raw, "party")
        color = palette_map.get(label, "#888888")
        df_p = df_c[df_c["party"] == party_raw].copy()
        df_p = df_p[df_p["condition"].isin(conditions)]
        df_p["_x"] = df_p["condition"].map(cond_to_x)
        df_p = df_p.sort_values("_x")

        if df_p.empty:
            continue

        # Small horizontal jitter so overlapping error bars are readable.
        jitter = (idx - (n_parties - 1) / 2.0) * 0.06
        x_vals = df_p["_x"].to_numpy() + jitter
        y_vals = df_p["rate"].to_numpy()
        yerr_low = (df_p["rate"] - df_p["ci_low"]).to_numpy()
        yerr_high = (df_p["ci_high"] - df_p["rate"]).to_numpy()

        ax.errorbar(
            x_vals,
            y_vals,
            yerr=[yerr_low, yerr_high],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.2,
            capsize=3,
            label=label,
        )

    # ── x-axis: a touch of padding so jittered end-points are not clipped ────
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_xticks(list(cond_to_x.values()))
    ax.set_xticklabels(conditions, rotation=0)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{subquestion_label} ({country})", pad=8)

    # ── Legend outside-right: one column, all country parties listed cleanly ──
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Party",
        handlelength=1.2,
        labelspacing=0.4,
    )

    # Reserve right margin for the legend before tight_layout so it is not clipped.
    fig.subplots_adjust(right=0.78)
    return fig


def _countries_in(df: pd.DataFrame) -> list[str]:
    """Unique countries present in df, DE first then the rest alphabetically.

    Args:
        df: Tidy scores table with a ``country`` column.

    Returns:
        Sorted list of country codes; ``"DE"`` is always first if present.
    """
    all_countries = sorted(df["country"].unique().tolist())
    if "DE" in all_countries:
        return ["DE"] + [c for c in all_countries if c != "DE"]
    return all_countries


def notebook_focus_slug(notebook_file: str | Path) -> str:
    """Convert notebook filename to folder slug used under ``analysis/figures``.

    Example:
        ``01_model_performance.ipynb`` -> ``model_performance``.
    """

    notebook_path = Path(notebook_file)
    stem_without_prefix = NOTEBOOK_PREFIX_PATTERN.sub("", notebook_path.stem).strip("_-")
    lowered = stem_without_prefix.lower()
    sanitized = NON_ALNUM_PATTERN.sub("_", lowered).strip("_")
    return sanitized or "notebook"


def plot_name_slug(plot_name: str) -> str:
    """Normalize a caller-supplied plot name into a filesystem-safe slug."""

    lowered = plot_name.lower().strip()
    normalized = NON_ALNUM_PATTERN.sub("_", lowered).strip("_")
    return normalized or "plot"


def _plot_overview_absolute(
    df: pd.DataFrame,
    *,
    names: DisplayNames,
    fig_width: float | None,
    aggregate: bool,
) -> plt.Figure:
    """Render a single absolute-mode overview figure.

    Args:
        df: Tidy scores table (may be pre-filtered to one country or the full
            multi-country frame when ``aggregate=True``).
        names: Display-name registry.
        fig_width: Total figure width in inches; ``None`` uses auto-sizing.
        aggregate: When ``True``, compute equal-country-weight means: average
            per-country ``adherence_index`` and ``robustness_score`` over
            countries before building the pivot. When ``False``, pass the frame
            directly (caller is responsible for country pre-filtering).

    Returns:
        The composed matplotlib Figure.
    """
    from src.analysis.aggregate import adherence_index, robustness_score

    # ── 1. Data assembly ────────────────────────────────────────────────────
    if aggregate:
        # Equal-country-weight: compute per (model, country, ie), then average
        # over countries → per (model, ie).
        per_model_country_ie = adherence_index(df, by=["model", "country", "ie"])
        ie_table = (
            per_model_country_ie.groupby(["model", "ie"], observed=True)["adherence"]
            .mean()
            .reset_index()
        )
        # Difficulty: mean over models of the equal-weight per-(model,ie) adherence.
        difficulty = ie_table.groupby("ie")["adherence"].mean()

        # Robustness: per (model, country), then average over countries → per model.
        per_model_country_rob = robustness_score(df, by=("model", "country"))
        rob_agg = (
            per_model_country_rob.groupby("model", observed=True)["robustness"]
            .mean()
            .reset_index()
            .set_index("model")["robustness"]
        )
    else:
        ie_table = adherence_index(df, by=["model", "country", "ie"])
        difficulty = ie_table.groupby("ie")["adherence"].mean()
        rob_agg_df = robustness_score(df, by=["model", "country"]).set_index("model")["robustness"]
        rob_agg = rob_agg_df

    non_baseline = [ie for ie in difficulty.index if ie != "baseline"]
    # Baseline first, then easiest → hardest (descending mean adherence)
    ordered_ies = (["baseline"] if "baseline" in difficulty.index else []) + sorted(
        non_baseline, key=lambda ie: difficulty[ie], reverse=True
    )

    pivot = ie_table.pivot(index="model", columns="ie", values="adherence")[ordered_ies]

    pivot.index = [names.apply(m, "model") for m in pivot.index]
    pivot.columns = [names.apply(ie, "ie") for ie in pivot.columns]

    rob = rob_agg.copy()
    rob.index = [names.apply(m, "model") for m in rob.index]
    rob = rob.reindex(pivot.index)  # match heatmap row order

    diff_display = difficulty.reindex(ordered_ies)
    diff_display.index = list(pivot.columns)

    n_models = len(pivot)
    n_ies = len(ordered_ies)

    return _build_overview_figure(
        pivot=pivot,
        rob=rob,
        diff_display=diff_display,
        n_models=n_models,
        n_ies=n_ies,
        fig_width=fig_width,
        cmap="viridis",
        diverging=False,
        title="Model × IE adherence  (baseline, then easiest → hardest)",
    )


def _plot_overview_delta(
    df: pd.DataFrame,
    *,
    names: DisplayNames,
    fig_width: float | None,
) -> plt.Figure:
    """Render a single delta-mode overview figure (NL − DE).

    The heatmap shows NL−DE deltas (negative = NL underperforms). The right
    robustness strip and bottom IE-difficulty strip both show NL absolute
    values.

    Args:
        df: Tidy scores table with exactly 2 countries.
        names: Display-name registry.
        fig_width: Total figure width in inches; ``None`` uses auto-sizing.

    Returns:
        The composed matplotlib Figure.
    """
    from src.analysis.aggregate import adherence_index, robustness_score
    from src.analysis.replication import country_delta, ie_difficulty_correlation

    present = _countries_in(df)
    country_a, country_b = present[0], present[1]
    df_a = df[df["country"] == country_a]
    df_b = df[df["country"] == country_b]

    # Heatmap cells: NL−DE delta pivoted model × ie.
    delta_df = country_delta(df, by=["model", "ie"])

    # Compute NL-based IE ordering (baseline first, then NL-easiest → hardest).
    nl_ie_adh = adherence_index(df_b, by=["model", "country", "ie"])
    nl_difficulty = nl_ie_adh.groupby("ie")["adherence"].mean()
    non_baseline = [ie for ie in nl_difficulty.index if ie != "baseline"]
    ordered_ies = (["baseline"] if "baseline" in nl_difficulty.index else []) + sorted(
        non_baseline, key=lambda ie: nl_difficulty[ie], reverse=True
    )

    # Build pivot from the delta frame.
    pivot = delta_df.pivot(index="model", columns="ie", values="delta")
    # Reorder columns by NL difficulty ordering (only keep IEs that exist in pivot).
    ordered_ies_filtered = [ie for ie in ordered_ies if ie in pivot.columns]
    pivot = pivot[ordered_ies_filtered]

    pivot.index = [names.apply(m, "model") for m in pivot.index]
    pivot.columns = [names.apply(ie, "ie") for ie in pivot.columns]

    # Primary sidebar values = NL absolute (bars).
    rob = robustness_score(df_b, by=["model", "country"]).set_index("model")["robustness"]
    rob.index = [names.apply(m, "model") for m in rob.index]
    rob = rob.reindex(pivot.index)

    diff_display = nl_difficulty.reindex(ordered_ies_filtered)
    diff_display.index = list(pivot.columns)

    # Secondary sidebar values = DE absolute (annotated alongside NL for comparison).
    de_ie_adh = adherence_index(df_a, by=["model", "country", "ie"])
    de_difficulty = de_ie_adh.groupby("ie")["adherence"].mean()
    diff_de = de_difficulty.reindex(ordered_ies_filtered)
    diff_de.index = list(pivot.columns)

    rob_de_raw = robustness_score(df_a, by=["model", "country"]).set_index("model")["robustness"]
    rob_de_raw.index = [names.apply(m, "model") for m in rob_de_raw.index]
    rob_de = rob_de_raw.reindex(pivot.index)

    # Optional: annotate overall IE-difficulty ρ in the title.
    try:
        corr = ie_difficulty_correlation(df, country_a=country_a, country_b=country_b)
        rho_note = f" | IE-difficulty ρ={corr.rho:.2f}"
    except Exception:
        rho_note = ""

    title = f"Model × IE  Deviation from {country_a} ({country_b}−{country_a}){rho_note}"

    n_models = len(pivot)
    n_ies = len(ordered_ies_filtered)

    return _build_overview_figure(
        pivot=pivot,
        rob=rob,
        diff_display=diff_display,
        n_models=n_models,
        n_ies=n_ies,
        fig_width=fig_width,
        cmap="RdBu_r",
        diverging=True,
        title=title,
        rob_label=f"Robustness SD ↓  ({country_b} bar · {country_a} annotated)",
        diff_label=f"IE difficulty  ({country_b} bar · {country_a} annotated)",
        annot_fmt="+.2f",
        rob_secondary=rob_de,
        diff_secondary=diff_de,
    )


def _build_overview_figure(
    *,
    pivot: pd.DataFrame,
    rob: pd.Series,
    diff_display: pd.Series,
    n_models: int,
    n_ies: int,
    fig_width: float | None,
    cmap: str,
    diverging: bool,
    title: str,
    rob_label: str = "Robustness SD ↓",
    diff_label: str = "IE difficulty\n(mean adherence)",
    annot_fmt: str = ".2f",
    rob_secondary: pd.Series | None = None,
    diff_secondary: pd.Series | None = None,
) -> plt.Figure:
    """Shared figure assembly for both absolute and delta overview modes.

    Args:
        pivot: Model × IE pivot table (display names already applied).
        rob: Robustness values indexed by model display name.
        diff_display: IE difficulty values indexed by IE display name.
        n_models: Number of rows in the heatmap.
        n_ies: Number of columns in the heatmap.
        fig_width: Total figure width in inches; ``None`` uses auto-sizing.
        cmap: Colormap name (e.g. ``"viridis"`` or ``"RdBu_r"``).
        diverging: If ``True``, use a symmetric diverging norm centered at 0.
        title: Figure title for the heatmap axis.
        rob_label: X-axis label for the robustness strip.
        diff_label: Y-axis label for the difficulty strip.
        annot_fmt: Format string for cell annotations.

    Returns:
        The composed matplotlib Figure.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm

    # ── Layout constants ─────────────────────────────────────────────────────
    ROB_WIDTH = 2.7 if rob_secondary is not None else 1.8  # inches for robustness strip
    CBAR_WIDTH = 0.55  # inches for colorbar strip
    DIFF_HEIGHT = 1.6  # inches for difficulty strip
    _fig_width = fig_width or (1.15 * n_ies + 1.5 + ROB_WIDTH + CBAR_WIDTH)
    _heatmap_height = 0.7 * n_models + 1.4
    _total_height = _heatmap_height + DIFF_HEIGHT

    # Proportional column widths
    _heat_w = _fig_width - ROB_WIDTH - CBAR_WIDTH
    _width_ratios = [_heat_w / _fig_width, ROB_WIDTH / _fig_width, CBAR_WIDTH / _fig_width]
    _height_ratios = [_heatmap_height / _total_height, DIFF_HEIGHT / _total_height]

    fig = plt.figure(figsize=(_fig_width, _total_height))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=_width_ratios,
        height_ratios=_height_ratios,
        hspace=0.08,
        wspace=0.06,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_rob = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_corner = fig.add_subplot(gs[1, 1:])  # merge bottom-right for clean look

    # ── Normalisation ────────────────────────────────────────────────────────
    heat_vals = pivot.values[~pd.isna(pivot.values)]
    if diverging:
        _raw_max = (
            max(abs(float(heat_vals.min())), abs(float(heat_vals.max()))) if heat_vals.size else 0.0
        )
        # TwoSlopeNorm requires vmin < vcenter < vmax; guard against all-zero deltas.
        abs_max = _raw_max if _raw_max > 1e-9 else 1e-6
        norm: Normalize = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
        vmin_arg: float | None = None
        vmax_arg: float | None = None
    else:
        vmin_arg = float(heat_vals.min()) if heat_vals.size else 0.0
        vmax_arg = 1.0
        norm = Normalize(vmin=vmin_arg, vmax=vmax_arg)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    sns.heatmap(
        pivot,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax_heat,
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_title(title, pad=8)
    plt.setp(ax_heat.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    plt.setp(ax_heat.get_yticklabels(), fontsize=8)

    # ── Robustness strip (right) ─────────────────────────────────────────────
    # y-positions: center of each heatmap row (0 = top in heatmap coords)
    y_centers = [i + 0.5 for i in range(n_models)]
    rob_vals = rob.values.astype(float)
    _rob_finite = rob_vals[~pd.isna(rob_vals)]
    rob_max = float(_rob_finite.max()) if _rob_finite.size else 1.0

    # Color each bar by its robustness value: map to viridis inverted (stable = dark)
    rob_norm = Normalize(vmin=0, vmax=rob_max * 1.1)
    rob_colors = plt.get_cmap("YlOrRd")(rob_norm(rob_vals))

    ax_rob.barh(
        y_centers,
        rob_vals,
        color=rob_colors,
        height=0.65,
        edgecolor="white",
        linewidth=0.4,
    )
    _sec_rob = rob_secondary.values.astype(float) if rob_secondary is not None else None
    for idx, (y, v) in enumerate(zip(y_centers, rob_vals, strict=False)):
        if not pd.isna(v):
            if _sec_rob is not None and not pd.isna(_sec_rob[idx]):
                label = f"{v:.3f} ({v - _sec_rob[idx]:+.3f})"
            else:
                label = f"{v:.3f}"
            ax_rob.text(v + 0.001, y, label, va="center", ha="left", fontsize=7, clip_on=False)

    ax_rob.set_ylim(n_models, 0)  # match heatmap: top row = index 0
    ax_rob.set_yticks([])
    ax_rob.set_xlabel(rob_label, fontsize=8)
    ax_rob.set_title("Robustness\n(lower = stable)", pad=8, fontsize=8)
    ax_rob.spines[["top", "right", "left"]].set_visible(False)
    ax_rob.tick_params(axis="x", labelsize=7)

    # ── IE difficulty strip (bottom) ─────────────────────────────────────────
    x_centers = [i + 0.5 for i in range(n_ies)]
    diff_vals = diff_display.values.astype(float)
    # Difficulty strip always coloured by the viridis/absolute scale, even in delta mode.
    diff_color_norm = Normalize(vmin=0.0, vmax=1.0)
    diff_cmap = "viridis"
    diff_colors = plt.get_cmap(diff_cmap)(diff_color_norm(diff_vals))

    ax_diff.bar(
        x_centers,
        diff_vals,
        color=diff_colors,
        width=0.65,
        edgecolor="white",
        linewidth=0.4,
    )
    _sec_diff = diff_secondary.values.astype(float) if diff_secondary is not None else None
    for idx, (x, v) in enumerate(zip(x_centers, diff_vals, strict=False)):
        if not pd.isna(v):
            if _sec_diff is not None and not pd.isna(_sec_diff[idx]):
                label = f"{v:.2f} ({v - _sec_diff[idx]:+.2f})"
            else:
                label = f"{v:.2f}"
            ax_diff.text(x, v + 0.01, label, ha="center", va="bottom", fontsize=7, clip_on=False)

    ax_diff.set_xlim(0, n_ies)
    ax_diff.set_xticks(x_centers)
    ax_diff.set_xticklabels(list(diff_display.index), rotation=25, ha="right", fontsize=8)
    ax_diff.set_ylabel(diff_label, fontsize=8)
    ax_diff.set_ylim(0, 1.1)
    ax_diff.spines[["top", "right"]].set_visible(False)
    ax_diff.tick_params(axis="y", labelsize=7)

    # ── Colorbar (top-right slot) ────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label("Adherence index" if not diverging else "Δ Adherence (NL−DE)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ── Corner cleanup ───────────────────────────────────────────────────────
    ax_corner.set_visible(False)

    return fig


# ── §3 Even-Handedness — appendix publication wrappers ───────────────────────


def plot_baseline_party_publication(
    df: pd.DataFrame,
    *,
    names: DisplayNames | None = None,
    logo_ticks: bool = False,
) -> plt.Figure:
    """Full-width DE + NL side-by-side baseline model × party adherence (appendix).

    Two panels sharing a single viridis colourbar fixed at
    ``[ADHERENCE_VMIN, ADHERENCE_VMAX]``. Absolute-inch layout with square cells.
    Purely reuses the exploratory ``_plot_party_adherence_heatmap_pub``; no
    exploratory function is modified.

    Args:
        df: Tidy scores table (both countries).
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        logo_ticks: Replace model y-axis labels with logo+name units.

    Returns:
        Full-width ``Figure`` for ``save_final_plot``.
    """
    from src.analysis.aggregate import adherence_index
    from src.analysis.display_names import DISPLAY_NAMES, PARTY_ORDER

    if names is None:
        names = DISPLAY_NAMES

    cell_in = 0.38
    left_pad = 0.48 if logo_ticks else 0.70
    mid_gap = 0.30
    right_pad = 0.10
    cbar_gap = 0.12
    cbar_w = 0.14
    top_pad = 0.35
    bot_pad = 0.70

    for country in ("DE", "NL"):
        sub = df[(df["country"] == country) & (df["ie"] == "baseline")]
        ai = adherence_index(sub, by=["model", "party"])
        if country == "DE":
            present_de = set(ai["party"].unique())
            n_party_de = sum(1 for p in PARTY_ORDER if p in present_de)
            n_model_de = ai["model"].nunique()
        else:
            present_nl = set(ai["party"].unique())
            n_party_nl = sum(1 for p in PARTY_ORDER if p in present_nl)
            n_model_nl = ai["model"].nunique()

    heat_w_de = n_party_de * cell_in
    heat_h = max(n_model_de, n_model_nl) * cell_in
    heat_w_nl = n_party_nl * cell_in

    fig_w = left_pad + heat_w_de + mid_gap + heat_w_nl + cbar_gap + cbar_w + right_pad
    fig_h = top_pad + heat_h + bot_pad

    def _r(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_de = fig.add_axes(_r(left_pad, bot_pad, heat_w_de, heat_h))
    ax_nl = fig.add_axes(_r(left_pad + heat_w_de + mid_gap, bot_pad, heat_w_nl, heat_h))
    cax = fig.add_axes(_r(
        left_pad + heat_w_de + mid_gap + heat_w_nl + cbar_gap,
        bot_pad, cbar_w, heat_h,
    ))

    common: dict = dict(
        ie="baseline", flag_threshold=None, names=names,
        vmin=ADHERENCE_VMIN, vmax=ADHERENCE_VMAX, cbar=False, rasterized=True,
    )
    _plot_party_adherence_heatmap_pub(df, country="DE", ax=ax_de, **common)
    _plot_party_adherence_heatmap_pub(df, country="NL", ax=ax_nl, **common)
    ax_nl.set_ylabel("")
    ax_nl.set_yticklabels([])

    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors
    sm = _cm.ScalarMappable(
        cmap=SEQUENTIAL_CMAP,
        norm=_mcolors.Normalize(vmin=ADHERENCE_VMIN, vmax=ADHERENCE_VMAX),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    _style_adherence_cbar(cb)

    ax_de.set_title("Germany", pad=6)
    ax_nl.set_title("Netherlands", pad=6)

    if logo_ticks:
        from src.analysis.display_names import MODEL_ORDER
        fig.canvas.draw()
        add_model_logo_ticks(ax_de, MODEL_ORDER, names)

    return fig


def _plot_party_adherence_heatmap_pub(
    df: pd.DataFrame,
    *,
    country: str,
    ie: str = "baseline",
    flag_threshold: float | None = None,
    names: DisplayNames | None = None,
    vmin: float = ADHERENCE_VMIN,
    vmax: float = ADHERENCE_VMAX,
    cbar: bool = False,
    rasterized: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Publication-only variant of plot_party_adherence_heatmap.

    Fixed vmin/vmax, percent annotations, cell-border flags, rasterized. Does
    NOT modify the exploratory function.
    """
    from src.analysis.aggregate import FLAG_THRESHOLD, adherence_index, rubric_dispersion
    from src.analysis.display_names import MODEL_ORDER, PARTY_ORDER

    if flag_threshold is None:
        flag_threshold = FLAG_THRESHOLD
    if names is None:
        from src.analysis.display_names import DISPLAY_NAMES
        names = DISPLAY_NAMES

    df_ie = df[(df["ie"] == ie) & (df["country"] == country)]
    adh = adherence_index(df_ie, by=["model", "party"])
    rdisp = rubric_dispersion(df_ie, by=["model", "party"])
    merged = adh.merge(rdisp, on=["model", "party"], how="left")
    merged["model_display"] = merged["model"].map(lambda m: names.apply(m, "model"))
    merged["party_display"] = merged["party"].map(lambda p: names.apply(p, "party"))

    adh_pivot = merged.pivot_table(
        index="model_display", columns="party_display", values="adherence"
    )
    spread_pivot = merged.pivot_table(
        index="model_display", columns="party_display", values="rubric_spread"
    )

    party_display_order = [names.apply(p, "party") for p in PARTY_ORDER]
    ordered_cols = [c for c in party_display_order if c in adh_pivot.columns]
    ordered_cols += sorted(c for c in adh_pivot.columns if c not in ordered_cols)
    adh_pivot = adh_pivot[ordered_cols]
    spread_pivot = spread_pivot[ordered_cols]

    model_display_order = [names.apply(m, "model") for m in MODEL_ORDER]
    ordered_rows = [r for r in model_display_order if r in adh_pivot.index]
    ordered_rows += sorted(r for r in adh_pivot.index if r not in ordered_rows)
    adh_pivot = adh_pivot.reindex(ordered_rows)
    spread_pivot = spread_pivot.reindex(ordered_rows)

    annot = adh_pivot.copy().astype(object)
    for row in adh_pivot.index:
        for col in adh_pivot.columns:
            val = adh_pivot.loc[row, col]
            annot.loc[row, col] = "" if pd.isna(val) else f"{round(float(val) * 100)}%"

    if ax is None:
        _, ax = plt.subplots()

    sns.heatmap(
        adh_pivot, annot=annot.values, fmt="",
        cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor="white", cbar=cbar, ax=ax,
        annot_kws={"fontsize": ANNOT_FONTSIZE},
    )
    for collection in ax.collections:
        collection.set_rasterized(rasterized)

    _flag_cell_borders(ax, spread_pivot >= flag_threshold)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ax


def plot_party_sq_breakdown_publication(
    deviation: pd.DataFrame,
    *,
    subquestion: str,
    country: str = "DE",
    names: DisplayNames | None = None,
    logo_ticks: bool = False,
) -> plt.Figure:
    """Full-width per-IE × model breakdown of one sub-question's party deviations (appendix).

    Three model panels at ``TEXT_WIDTH_IN``. House-style cmap (``DIVERGING_CMAP``)
    and cell-border flags. The exploratory ``plot_party_sq_breakdown`` is NOT modified.

    Args:
        deviation: Per-context deviation frame (``party_sq_deviation`` output).
        subquestion: Raw sub-question id.
        country: Country code (default ``"DE"``).
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        logo_ticks: Replace party y-axis labels with logo icons (first panel only).

    Returns:
        Full-width ``Figure``.
    """
    from src.analysis.aggregate import robust_symmetric_vmax
    from src.analysis.display_names import DISPLAY_NAMES, MODEL_ORDER, PARTY_ORDER

    if names is None:
        names = DISPLAY_NAMES

    sub = deviation[
        (deviation["country"] == country) & (deviation["subquestion"] == subquestion)
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows for country={country!r}, subquestion={subquestion!r}")

    ie_canonical = ["baseline", "noise", "prior_conflict", "clarity", "availability", "consistency"]
    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    if not models:
        models = sorted(sub["model"].unique())

    present_parties = set(sub["party"])
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    party_labels = [names.apply(p, "party") for p in party_order]
    present_ie = set(sub["ie"])
    ie_order = [ie for ie in ie_canonical if ie in present_ie]
    ie_labels = [names.apply(ie, "ie") for ie in ie_order]

    sub["party_label"] = sub["party"].map(lambda p: names.apply(p, "party"))
    sub["ie_label"] = sub["ie"].map(lambda i: names.apply(i, "ie"))

    vmax = robust_symmetric_vmax(sub["deviation"])

    n = len(models)

    # Absolute-inch layout — identical cell rectangles for all model panels.
    # Seaborn's built-in cbar steals width from the last panel; placing axes
    # manually and using a separate cax avoids that entirely.
    cell_w: float = 0.55   # inches per IE column
    cell_h: float = 0.28   # inches per party row
    n_ie = len(ie_labels)
    n_party = len(party_labels)
    heat_w = n_ie * cell_w
    heat_h = n_party * cell_h

    left_pad: float = 0.42 if logo_ticks else 0.82   # party-label room on the first panel
    panel_gap: float = 0.22  # gap between model panels
    cbar_gap: float = 0.14
    cbar_w: float = 0.18
    right_pad: float = 0.05
    top_pad: float = 0.45    # title clearance
    bot_pad: float = 0.92    # rotated x-tick label clearance

    fig_w = left_pad + n * heat_w + (n - 1) * panel_gap + cbar_gap + cbar_w + right_pad
    fig_h = top_pad + heat_h + bot_pad

    def _r(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    fig = plt.figure(figsize=(fig_w, fig_h))
    heatmap_axes: list[plt.Axes] = []
    for col in range(n):
        x0 = left_pad + col * (heat_w + panel_gap)
        heatmap_axes.append(fig.add_axes(_r(x0, bot_pad, heat_w, heat_h)))
    cax = fig.add_axes(_r(
        left_pad + n * heat_w + (n - 1) * panel_gap + cbar_gap,
        bot_pad, cbar_w, heat_h,
    ))

    for col, (model, ax) in enumerate(zip(models, heatmap_axes, strict=True)):
        m = sub[sub["model"] == model]
        dev_p = m.pivot_table(index="party_label", columns="ie_label", values="deviation").reindex(
            index=party_labels, columns=ie_labels
        )

        annot = dev_p.copy().astype(object)
        for row in dev_p.index:
            for c in dev_p.columns:
                val = dev_p.loc[row, c]
                if pd.isna(val) or abs(float(val) * 100) < 1.0:
                    annot.loc[row, c] = ""
                    continue
                annot.loc[row, c] = f"{float(val) * 100:+.0f}"

        sns.heatmap(
            dev_p, annot=annot.values, fmt="",
            cmap=DIVERGING_CMAP, center=0.0, vmin=-vmax, vmax=vmax,
            linewidths=0.5, linecolor="white",
            annot_kws={"fontsize": ANNOT_FONTSIZE},
            cbar=False,
            ax=ax,
        )
        for collection in ax.collections:
            collection.set_rasterized(True)

        ax.set_title(names.apply(model, "model"))
        ax.set_xlabel("")
        # hide y-axis on panels after the first
        if col == 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ie_mean = m.groupby("ie_label", observed=True)["rate"].mean()
        ax.set_xticklabels(
            [f"{ie}\nμ={ie_mean.get(ie, float('nan')):.2f}" for ie in ie_labels],
            rotation=30, ha="right",
        )

    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors
    sm = _cm.ScalarMappable(
        cmap=DIVERGING_CMAP,
        norm=_mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Deviation")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(direction="in", length=2, width=0.5)

    if logo_ticks:
        fig.canvas.draw()
        add_party_logo_ticks(heatmap_axes[0], party_order, names)

    return fig


def plot_ablation_pair(
    delta_df: pd.DataFrame,
    *,
    contrast_label: str,
    delta_only: bool = False,
    names: DisplayNames | None = None,
    logo_ticks: bool = False,
) -> plt.Figure:
    """Full-width DE + NL ablation delta figure (appendix, ``figure*``).

    Two layout modes:

    - ``delta_only=False`` (legacy): four panels (rate DE, delta DE, rate NL,
      delta NL) in a 2×2 grid; rows share y-axes, columns share x-axes.
    - ``delta_only=True``: two panels side-by-side (delta DE | delta NL) with
      independent party axes (each country's own party labels on the left),
      sized at ``TEXT_WIDTH_IN``. This is the intended publication mode —
      the delta is the signal; the pass-rate panel is redundant appendix material.

    Shared delta ``vmax`` is computed across both countries in both modes.
    Rubric-block separators match the H2 party × SQ style.

    Args:
        delta_df: Full multi-country ``paired_delta`` output.
        contrast_label: Delta title prefix (e.g. ``r"$\\Delta$ anon $-$ full (Baseline)"``).
        delta_only: When ``True``, emit the compact 1×2 delta-only layout.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.
        logo_ticks: Replace party y-axis labels with logo icons (delta_only only).

    Returns:
        Full-width ``Figure``.
    """
    from src.analysis.aggregate import robust_symmetric_vmax
    from src.analysis.display_names import (
        DISPLAY_NAMES,
        PARTY_ORDER,
        SUBQUESTION_ORDER,
        SUBQUESTION_RUBRIC,
    )

    if names is None:
        names = DISPLAY_NAMES

    present_sq = set(delta_df["subquestion"].unique())
    sq_order = [sq for sq in SUBQUESTION_ORDER if sq in present_sq]
    sq_labels = [names.apply(sq, "subquestion") for sq in sq_order]
    delta_vmax = robust_symmetric_vmax(delta_df["delta_rate"])
    rubric_seq = [SUBQUESTION_RUBRIC[sq] for sq in sq_order]

    def _separators(ax: plt.Axes) -> None:
        for i in range(1, len(rubric_seq)):
            if rubric_seq[i] != rubric_seq[i - 1]:
                ax.axvline(i, color="#333333", lw=1.4)

    def _draw_rate(ax: plt.Axes, pivot: pd.DataFrame, title: str) -> None:
        annot = pivot.copy().astype(object)
        for row in pivot.index:
            for col in pivot.columns:
                val = pivot.loc[row, col]
                annot.loc[row, col] = "" if pd.isna(val) else f"{int(round(float(val) * 100))}"
        sns.heatmap(pivot, annot=annot.values, fmt="",
                    cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Pass rate", "shrink": 0.85}, ax=ax)
        for collection in ax.collections:
            collection.set_rasterized(True)
        _separators(ax)
        ax.set_title(title, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    def _draw_delta(ax: plt.Axes, pivot: pd.DataFrame, title: str, *, show_cbar: bool = True) -> None:
        annot = _build_delta_annot(pivot)
        cbar_kws = {"label": r"$\Delta$ pass rate", "shrink": 0.85} if show_cbar else None
        sns.heatmap(pivot, annot=annot.values, fmt="",
                    cmap=DIVERGING_CMAP, center=0.0, vmin=-delta_vmax, vmax=delta_vmax,
                    linewidths=0.5, linecolor="white",
                    cbar=show_cbar, cbar_kws=cbar_kws, ax=ax)
        for collection in ax.collections:
            collection.set_rasterized(True)
        _separators(ax)
        ax.set_title(title, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    if delta_only:
        # ── 1×2 compact delta-only layout ────────────────────────────────────
        # Each country's party labels stay on its own left axis — different
        # n_parties per country (DE=7, NL=8) so sharey would mismatch.
        def _build_delta(country: str) -> tuple[pd.DataFrame, list[str]]:
            present = set(delta_df[delta_df["country"] == country]["party"].unique())
            p_order = [p for p in PARTY_ORDER if p in present]
            p_labels = [names.apply(p, "party") for p in p_order]
            df_c = delta_df[delta_df["country"] == country].copy()
            df_c["_pl"] = df_c["party"].map(lambda p: names.apply(p, "party"))
            df_c["_sl"] = df_c["subquestion"].map(lambda s: names.apply(s, "subquestion"))
            delta_p = df_c.pivot_table(
                index="_pl", columns="_sl", values="delta_rate"
            ).reindex(index=p_labels, columns=sq_labels)
            return delta_p, p_labels

        delta_de, party_labels_de = _build_delta("DE")
        delta_nl, party_labels_nl = _build_delta("NL")

        n_sq = len(sq_order)
        # Cell size: +10% vs previous 0.38×0.30 → wider, slightly taller cells.
        cell_w = 0.42
        cell_h = 0.33
        left_lbl_w = 0.42 if logo_ticks else 1.1   # room for party labels / logos
        cbar_w = 0.25
        cbar_gap = 0.12
        mid_gap = 0.55      # gap between DE and NL panels
        top_pad = 0.35
        bot_pad = 0.80

        heat_w = n_sq * cell_w
        heat_h_de = len(party_labels_de) * cell_h
        heat_h_nl = len(party_labels_nl) * cell_h
        heat_h = max(heat_h_de, heat_h_nl)

        fig_w = left_lbl_w + heat_w + mid_gap + left_lbl_w + heat_w + cbar_gap + cbar_w
        fig_h = top_pad + heat_h + bot_pad

        def _r(x: float, y: float, w: float, h: float) -> list[float]:
            return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

        fig = plt.figure(figsize=(fig_w, fig_h))
        ax_de = fig.add_axes(_r(left_lbl_w, bot_pad, heat_w, heat_h_de))
        ax_nl = fig.add_axes(_r(left_lbl_w + heat_w + mid_gap + left_lbl_w, bot_pad, heat_w, heat_h_nl))
        cax = fig.add_axes(_r(
            left_lbl_w + heat_w + mid_gap + left_lbl_w + heat_w + cbar_gap,
            bot_pad, cbar_w, heat_h,
        ))

        _draw_delta(ax_de, delta_de, f"{contrast_label} — DE", show_cbar=False)
        _draw_delta(ax_nl, delta_nl, f"{contrast_label} — NL", show_cbar=False)

        import matplotlib.cm as _cm
        import matplotlib.colors as _mcolors
        sm = _cm.ScalarMappable(
            cmap=DIVERGING_CMAP,
            norm=_mcolors.TwoSlopeNorm(vmin=-delta_vmax, vcenter=0.0, vmax=delta_vmax),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(r"$\Delta$ pass rate")
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(direction="in", length=2, width=0.5)

        if logo_ticks:
            # _build_delta returns p_order already filtered to present parties
            de_party_order = [p for p in PARTY_ORDER if p in set(delta_df[delta_df["country"] == "DE"]["party"].unique())]
            nl_party_order = [p for p in PARTY_ORDER if p in set(delta_df[delta_df["country"] == "NL"]["party"].unique())]
            fig.canvas.draw()
            add_party_logo_ticks(ax_de, de_party_order, names)
            add_party_logo_ticks(ax_nl, nl_party_order, names)

        return fig

    # ── legacy 2×2 layout ────────────────────────────────────────────────────
    present_parties = set(delta_df["party"].unique())
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    party_labels = [names.apply(p, "party") for p in party_order]

    def _build(country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_c = delta_df[delta_df["country"] == country].copy()
        df_c["_pl"] = df_c["party"].map(lambda p: names.apply(p, "party"))
        df_c["_sl"] = df_c["subquestion"].map(lambda s: names.apply(s, "subquestion"))
        rate_p = df_c.pivot_table(index="_pl", columns="_sl", values="rate_full").reindex(
            index=party_labels, columns=sq_labels)
        delta_p = df_c.pivot_table(index="_pl", columns="_sl", values="delta_rate").reindex(
            index=party_labels, columns=sq_labels)
        return rate_p, delta_p

    rate_de, delta_de = _build("DE")
    rate_nl, delta_nl = _build("NL")

    n_sq = len(sq_order)
    n_parties = len(party_labels)
    panel_w = 0.55 * n_sq + 1.8
    panel_h = 0.40 * n_parties + 1.8
    fig, axes = plt.subplots(2, 2, figsize=(panel_w * 2 + 1.0, panel_h * 2 + 0.5),
                             sharey="row", sharex="col")

    _draw_rate(axes[0][0], rate_de, "Pass rate — DE")
    _draw_delta(axes[0][1], delta_de, f"{contrast_label} (DE)")
    _draw_rate(axes[1][0], rate_nl, "Pass rate — NL")
    _draw_delta(axes[1][1], delta_nl, f"{contrast_label} (NL)")

    fig.tight_layout()
    return fig


def plot_ablation_slope_publication(
    slope_df: pd.DataFrame,
    *,
    subquestion_label: str,
    country: str,
    names: DisplayNames | None = None,
) -> plt.Figure:
    """Single-column sanitization slope chart at ``COLUMN_WIDTH_IN`` (appendix).

    Thin wrapper around ``plot_ablation_slope`` logic, enforcing the publication
    column width. The exploratory function is NOT modified.

    Args:
        slope_df: Tidy slope frame (country, party, condition, rate, ci_low, ci_high).
        subquestion_label: Axes title prefix (e.g. ``r"Sanitization (I4)"``).
        country: Country code.
        names: Display-name registry; defaults to ``DISPLAY_NAMES``.

    Returns:
        Single-column ``Figure``.
    """
    from src.analysis.display_names import DISPLAY_NAMES, PARTY_ORDER, party_palette

    if names is None:
        names = DISPLAY_NAMES

    df_c = slope_df[slope_df["country"] == country].copy()
    if df_c.empty:
        raise ValueError(f"No rows for country={country!r}")

    _COND_ORDER: list[str] = ["full", "anon", "english"]
    conditions = [c for c in _COND_ORDER if c in set(df_c["condition"].unique())]
    cond_to_x: dict[str, int] = {c: i for i, c in enumerate(conditions)}
    palette_map = party_palette(names)
    present_parties = set(df_c["party"].unique())
    party_order = [p for p in PARTY_ORDER if p in present_parties]
    n_parties = len(party_order)

    fig_h = 0.22 * n_parties + 1.2
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, fig_h))

    for idx, party_raw in enumerate(party_order):
        label = names.apply(party_raw, "party")
        color = palette_map.get(label, "#888888")
        df_p = df_c[df_c["party"] == party_raw].copy()
        df_p = df_p[df_p["condition"].isin(conditions)]
        df_p["_x"] = df_p["condition"].map(cond_to_x)
        df_p = df_p.sort_values("_x")
        if df_p.empty:
            continue
        jitter = (idx - (n_parties - 1) / 2.0) * 0.06
        ax.errorbar(
            df_p["_x"].to_numpy() + jitter,
            df_p["rate"].to_numpy(),
            yerr=[(df_p["rate"] - df_p["ci_low"]).to_numpy(),
                  (df_p["ci_high"] - df_p["rate"]).to_numpy()],
            color=color, marker="o", markersize=3, linewidth=1.0,
            capsize=2, label=label,
        )

    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_xticks(list(cond_to_x.values()))
    ax.set_xticklabels(conditions, rotation=0)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0.60, 1.02)
    # Legend inside the axes — bottom-right corner is empty (all lines converge
    # toward the left at 'Full', leaving the right side of the plot free).
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        edgecolor="#cccccc",
        title="Party",
        title_fontsize=ANNOT_FONTSIZE - 1,
        fontsize=ANNOT_FONTSIZE - 1,
        handlelength=0.9,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.5,
    )
    return fig

def _draw_ie_rubric_heatmap(
    df: pd.DataFrame,
    *,
    ie: str,
    flag_threshold: float,
    names: DisplayNames,
    ax: plt.Axes | None,
    rho_by_rubric_display: dict[str, str],
) -> plt.Axes:
    """Draw a single IE × rubric heatmap panel.

    Args:
        df: Tidy scores table (already filtered to the desired country if needed).
        ie: Raw IE id.
        flag_threshold: SQ-spread flag cutoff.
        names: Display-name registry.
        ax: Existing axis or ``None`` to create a new figure.
        rho_by_rubric_display: Mapping from display rubric name to ρ annotation
            string. Empty dict suppresses annotation.

    Returns:
        The ``Axes`` the heatmap was drawn on.
    """
    from src.analysis.aggregate import adherence_index, subquestion_dispersion

    df_ie = df[df["ie"] == ie]
    adh = adherence_index(df_ie, by=["model", "country", "rubric"])
    spread = subquestion_dispersion(df_ie, by=("model", "country", "rubric"))
    merged = adh.merge(spread, on=["model", "country", "rubric"], how="left")
    merged["model_display"] = merged["model"].map(lambda m: names.apply(m, "model"))
    merged["rubric_display"] = merged["rubric"].map(lambda r: names.apply(r, "rubric"))

    adh_pivot = merged.pivot(index="model_display", columns="rubric_display", values="adherence")
    spread_pivot = merged.pivot(index="model_display", columns="rubric_display", values="sq_spread")

    annot = adh_pivot.copy().astype(object)
    for row in adh_pivot.index:
        for col in adh_pivot.columns:
            value = adh_pivot.loc[row, col]
            if pd.isna(value):
                annot.loc[row, col] = ""
                continue
            sq_spread = spread_pivot.loc[row, col]
            flagged = pd.notna(sq_spread) and sq_spread >= flag_threshold
            annot.loc[row, col] = f"{value:.2f}" + ("*" if flagged else "")

    if ax is None:
        _, ax = plt.subplots(
            figsize=(1.5 * len(adh_pivot.columns) + 2.0, 0.7 * len(adh_pivot) + 1.6)
        )
    sns.heatmap(
        adh_pivot,
        annot=annot.values,
        fmt="",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Adherence index"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{names.apply(ie, 'ie')} — model x rubric  (* = SQ spread >= {flag_threshold:.2f})",
        pad=10,
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # Append ρ annotations to x-tick labels when available.
    if rho_by_rubric_display:
        new_labels = []
        for label in ax.get_xticklabels():
            display_name = label.get_text()
            rho_str = rho_by_rubric_display.get(display_name)
            if rho_str:
                new_labels.append(f"{display_name}\n{rho_str}")
            else:
                new_labels.append(display_name)
        ax.set_xticklabels(new_labels, rotation=20, ha="right")

    return ax
