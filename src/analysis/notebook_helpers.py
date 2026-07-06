"""Notebook-level display helpers: save + display + close wrappers.

Each function calls the appropriate plotting primitive, persists the figure
via save_plot / save_panels, shows it in the notebook, then closes it.
All notebook scope variables (df, REPO_ROOT, NOTEBOOK_FILE) are passed as
arguments so these helpers are reusable across notebooks without globals.

Usage in notebook setup cell::

    from src.analysis import show_ie_pair, show_sq_drill, show_sq_pair
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

from src.analysis.aggregate import subquestion_adherence
from src.analysis.display_names import DISPLAY_NAMES, DisplayNames
from src.analysis.plotting import (
    plot_country_delta_heatmap,
    plot_ie_rubric_heatmap,
    plot_subquestion_adherence,
    render_de_delta_pair,
    save_panels,
    save_plot,
)


def show_ie_pair(
    df: pd.DataFrame,
    ie: str,
    *,
    repo_root: Path,
    notebook_file: Path,
    show_country_correlation: bool = True,
) -> None:
    """DE heatmap + NL−DE delta for one IE; save as two files, display composite.

    Args:
        df: Full tidy scores frame (both countries).
        ie: IE condition ID (e.g. ``"noise"``, ``"consistency"``).
        repo_root: Repo root Path (``REPO_ROOT`` notebook variable).
        notebook_file: Notebook file Path (``NOTEBOOK_FILE`` notebook variable).
        show_country_correlation: Annotate Spearman ρ on the DE heatmap.
    """
    df_ie = df[df["ie"] == ie]
    n_rubrics = max(df_ie["rubric"].nunique(), 1)
    n_models = max(df_ie["model"].nunique(), 1)
    panel_size = (1.5 * n_rubrics + 2.0, 0.7 * n_models + 1.8)

    def _draw_de(ax: plt.Axes) -> None:
        plot_ie_rubric_heatmap(
            df, ie=ie, country="DE",
            show_country_correlation=show_country_correlation, ax=ax,
        )
        ax.set_title(f"(DE) {ax.get_title()}")

    def _draw_delta(ax: plt.Axes) -> None:
        plot_country_delta_heatmap(df, axis="rubric", ie=ie, star_from_nl_sq=True, ax=ax)

    fig_de, fig_delta, fig_combo = render_de_delta_pair(
        _draw_de, _draw_delta, panel_size=panel_size
    )
    saved = save_panels(
        fig_de=fig_de, fig_delta=fig_delta,
        repo_root=repo_root, notebook_file=notebook_file,
        plot_name=f"rubric_heatmap_{ie}", subfolder=ie,
    )
    print({k: str(p.relative_to(repo_root)) for k, p in saved.items()})
    display(fig_combo)
    plt.close("all")


def show_sq_drill(
    df: pd.DataFrame,
    ie: str,
    rubric: str,
    *,
    repo_root: Path,
    notebook_file: Path,
    names: DisplayNames | None = None,
    baseline_reference: bool = False,
    models: list[str] | None = None,
    plot_name: str | None = None,
) -> None:
    """Single IE × rubric sub-question drill: plot, save, display, close.

    Args:
        df: Tidy scores frame (typically pre-filtered to one country via ``df_de``).
        ie: IE condition ID to drill into.
        rubric: Rubric ID to drill into.
        repo_root: Repo root Path (``REPO_ROOT`` notebook variable).
        notebook_file: Notebook file Path (``NOTEBOOK_FILE`` notebook variable).
        names: DisplayNames registry (defaults to ``DISPLAY_NAMES``).
        baseline_reference: Show baseline pass-rate reference line.
        models: Optional model ID filter.
        plot_name: Output filename stem (default: ``sq_{ie}_{rubric}``).
    """
    ax = plot_subquestion_adherence(
        df, ie=ie, rubric=rubric,
        baseline_reference=baseline_reference,
        models=models,
        names=names or DISPLAY_NAMES,
    )
    ax.figure.tight_layout()
    saved = save_plot(
        fig=ax.figure, repo_root=repo_root, notebook_file=notebook_file,
        plot_name=plot_name or f"sq_{ie}_{rubric}",
    )
    print({k: str(p.relative_to(repo_root)) for k, p in saved.items()})
    display(ax.figure)
    plt.close(ax.figure)


def show_sq_pair(
    df: pd.DataFrame,
    rubric: str,
    *,
    repo_root: Path,
    ie: str | None = None,
    notebook_file: Path,
    names: DisplayNames | None = None,
    baseline_reference: bool | None = False,
    country_a: str = "DE",
    country_b: str = "NL",
) -> None:
    """Per-rubric SQ pass rates: country_a bars (left) + delta (right); two files.

    If ie is specified, restricts to that IE condition.
    Saved as ``sq_{rubric}_de.png`` and
    ``sq_{rubric}_delta.png`` via save_panels.

    Args:
        df: Full tidy scores frame (both countries).
        rubric: Rubric ID (e.g. ``"impartiality"``, ``"epistemic_calibration"``).
        ie: Optional IE condition ID to filter on.
        repo_root: Repo root Path (``REPO_ROOT`` notebook variable).
        notebook_file: Notebook file Path (``NOTEBOOK_FILE`` notebook variable).
        names: DisplayNames registry (defaults to ``DISPLAY_NAMES``).
        country_a: Reference country code (default ``"DE"``).
        country_b: Comparison country code (default ``"NL"``).
    """
    _names = names or DISPLAY_NAMES
    rubric_label = _names.apply(rubric, "rubric")

    df_filt = df[df["rubric"] == rubric]
    if ie is not None:
        df_filt = df_filt[df_filt["ie"] == ie]

    rates = subquestion_adherence(
        df_filt, by=("model", "subquestion", "country")
    )
    rates["sq"] = rates["subquestion"].str.replace("_", " ")
    wide = rates.pivot_table(
        index=["model", "sq"], columns="country", values="adherence"
    ).reset_index()
    wide.columns.name = None
    wide = wide.dropna(subset=[country_a, country_b])
    wide["delta"] = wide[country_b] - wide[country_a]
    wide["series"] = wide["model"].map(lambda m: _names.apply(m, "model"))

    order = wide.groupby("sq")[country_a].mean().sort_values().index.tolist()
    model_series = wide["series"].drop_duplicates().tolist()
    avg = wide.groupby("sq", as_index=False)["delta"].mean().assign(series="Average")
    delta_df = pd.concat([wide[["sq", "series", "delta"]], avg], ignore_index=True)
    hue_order = model_series + ["Average"]
    base_colors = sns.color_palette()
    palette: dict[str, object] = {m: base_colors[i] for i, m in enumerate(model_series)}
    palette["Average"] = "0.6"
    lim = float(delta_df["delta"].abs().max()) * 1.25 + 1e-6

    if ie is not None:
        plot_prefix = f"sq_{rubric}_{ie}"
        title_prefix = f"[{ie}] "
    else:
        plot_prefix = f"sq_{rubric}"
        title_prefix = ""

    def _draw_a(ax: plt.Axes) -> None:
        plot_subquestion_adherence(
            df, rubric=rubric, ie=ie, country=country_a, ax=ax, names=_names, baseline_reference=(baseline_reference or False),
            title=f"{title_prefix}({country_a}) {rubric_label} — sub-question adherence",
       
        )

    def _draw_delta(ax: plt.Axes) -> None:
        sns.barplot(
            data=delta_df, x="delta", y="sq", hue="series",
            order=order, hue_order=hue_order, palette=palette, orient="h", ax=ax,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%+.2f", padding=2, fontsize=7)
        ax.axvline(0, color="#333333", lw=1.0, zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(f"{country_b} − {country_a} pass-rate delta  (+ = {country_b} higher)")
        ax.set_ylabel("")
        ax.set_title(f"{title_prefix}{country_b}−{country_a} deviation — {rubric_label}", pad=8)
        ax.legend(title="", frameon=False, loc="lower right")
        sns.despine(ax=ax)

    panel_size = (7.0, 0.62 * len(order) + 2.0)
    fig_a, fig_delta, fig_combo = render_de_delta_pair(_draw_a, _draw_delta, panel_size=panel_size)
    saved = save_panels(
        fig_de=fig_a, fig_delta=fig_delta,
        repo_root=repo_root, notebook_file=notebook_file,
        plot_name=plot_prefix,
    )
    print({k: str(p.relative_to(repo_root)) for k, p in saved.items()})
    display(fig_combo)
    plt.close("all")
