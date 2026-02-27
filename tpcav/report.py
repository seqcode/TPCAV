from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import io
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Union

def _read_as_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _render_kv_table(rows: list[tuple[str, str]]) -> str:
    cells = "\n".join(
        f"<tr><th>{_html.escape(k)}</th><td>{_html.escape(v)}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{cells}</table>"


def _render_df_table(df: Any, max_rows: int = 200) -> str:
    try:
        df2 = df.head(max_rows)
        return df2.to_html(index=False, escape=True, classes="df")
    except Exception:
        return f"<pre>{_html.escape(str(df)[:20000])}</pre>"


def _json_for_html(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")


def _maybe_build_motif_logo_data_uris(
    motif_meme_file: Optional[Union[str, Path]], concept_names: list[str]
) -> dict[str, str]:
    if motif_meme_file is None or not concept_names:
        return {}
    try:
        from Bio import motifs
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import logomaker

        from . import utils as _utils
    except Exception:
        return {}

    motif_meme_file = Path(motif_meme_file)
    with motif_meme_file.open() as handle:
        motif_pwms = motifs.parse(handle, fmt="MINIMAL")
    pwm_by_name = {_utils.clean_motif_name(m.name): m.pwm for m in motif_pwms}

    def compute_ic(row: Any) -> float:
        entropy = -sum([p * np.log2(p) if p > 0 else 0 for p in row])
        return 2 - entropy

    out: dict[str, str] = {}
    for name in concept_names:
        pwm = pwm_by_name.get(_utils.clean_motif_name(str(name)))
        if pwm is None:
            continue
        pwm_df = pd.DataFrame(pwm)[["A", "C", "G", "T"]]
        ic_df = pwm_df.copy()
        for i in range(len(pwm_df)):
            ic_total = compute_ic(pwm_df.iloc[i])
            ic_df.iloc[i] = pwm_df.iloc[i] * ic_total

        fig, ax = plt.subplots(figsize=(2.2, 0.55))
        logomaker.Logo(
            ic_df,
            color_scheme={"A": "red", "C": "blue", "G": "orange", "T": "green"},
            ax=ax,
        )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        out[str(name)] = "data:image/png;base64," + base64.b64encode(
            buf.getvalue()
        ).decode("ascii")

    return out


def generate_tpcav_html_report(
    output_html_path: Union[str, Path],
    motif_cav_trainers: Optional[Mapping[Union[int, str], Any]] = None,
    extra_cav_trainers: Optional[Mapping[str, Any]] = None,
    attributions: Optional[Union[Any, list[Any]]] = None,
    motif_file: Optional[Union[str, Path]] = None,
    motif_file_fmt: str = "meme",
    fscore_thresh: float = 0.8,
    top_motif_concepts: int = 10,
    title: str = "TPCAV report",
    embed_images: bool = False,
) -> Path:
    """
    Generate a standalone HTML report from CavTrainer objects.

    Args:
        output_html_path: Path to write the HTML report.
        motif_cav_trainers: Dict mapping "# motif insertions" -> CavTrainer.
        extra_cav_trainers: Optional dict mapping a display name -> CavTrainer.
        attributions: Optional attribution tensor (or list of tensors) shared across all heatmaps (enables log-ratio barplots).
        motif_file: Motif file path passed through to compute/plot helpers when relevant.
        motif_file_fmt: 'meme' or 'consensus' (used by compute_motif_auc_fscore).
        fscore_thresh: Threshold passed to CavTrainer.plot_cavs_similaritiy_heatmap.
        title: Report title.
        embed_images: If True, embed PNGs as base64 data URIs; otherwise link to files.
    """
    output_html_path = Path(output_html_path)
    assets_dir = output_html_path.parent / f"{output_html_path.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    motif_cav_trainers = dict(motif_cav_trainers or {})
    extra_cav_trainers = dict(extra_cav_trainers or {})

    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    motif_auc_plot_path = assets_dir / "motif_auc_fscore_regression.png"
    motif_auc_df = None
    motif_auc_plot_exists = False
    ranked_motif_concepts: list[str] = []
    motif_insertions: list[int] = []

    if motif_cav_trainers:
        try:
            from .cavs import compute_motif_auc_fscore  # local import (torch dependency)
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import `compute_motif_auc_fscore` from `tpcav.cavs`. "
                "Ensure your environment can import torch and tpcav."
            ) from e

        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        try:
            sorted_keys = sorted(motif_cav_trainers.keys(), key=lambda x: int(x))
            motif_insertions = [int(k) for k in sorted_keys]
            trainers = [motif_cav_trainers[k] for k in sorted_keys]
            motif_auc_df = compute_motif_auc_fscore(
                num_motif_insertions=motif_insertions,
                cav_trainers=trainers,
                motif_file=str(motif_file) if motif_file is not None else None,
                motif_file_fmt=motif_file_fmt,
                output_path=str(motif_auc_plot_path),
            )
            motif_auc_plot_exists = motif_auc_plot_path.exists()
            if motif_auc_df is not None and "concept" in motif_auc_df.columns:
                ranked_motif_concepts = motif_auc_df["concept"].astype(str).tolist()
        finally:
            if plt is not None:
                plt.close("all")

    # -----------------------------------------------------------------------------
    # 1) Gather CavTrainers
    # -----------------------------------------------------------------------------
    all_trainers: dict[str, Any] = {f"motifs_{k}": v for k, v in motif_cav_trainers.items()}
    all_trainers.update(extra_cav_trainers)

    motif_rank: dict[str, int] = {c: i + 1 for i, c in enumerate(ranked_motif_concepts)}
    motif_concept_set = set(ranked_motif_concepts)

    motif_weight_source = None
    if motif_cav_trainers and motif_insertions:
        motif_weight_source = motif_cav_trainers[max(motif_insertions)]

    # -----------------------------------------------------------------------------
    # 2) Concept info table (motif first, then extras)
    # -----------------------------------------------------------------------------
    concept_rows: list[dict[str, Any]] = []

    if motif_auc_df is not None:
        for row in motif_auc_df.to_dict(orient="records"):
            concept = str(row.get("concept"))
            out: dict[str, Any] = {
                "concept": concept,
                "type": "motif",
                "rank": motif_rank.get(concept),
                "source": "motif",
            }
            for nm in motif_insertions:
                out[f"fscore_{nm}"] = row.get(nm)
            out["AUC_fscores"] = row.get("AUC_fscores")
            out["AUC_fscores_residual"] = row.get("AUC_fscores_residual")
            concept_rows.append(out)

    # Track all non-motif concepts (concept, source).
    extra_concepts: list[tuple[str, str, float]] = []
    for trainer_name, trainer in extra_cav_trainers.items():
        cav_fscores = getattr(trainer, "cav_fscores", None)
        if not isinstance(cav_fscores, dict):
            continue
        for concept, fscore in cav_fscores.items():
            concept = str(concept)
            if concept in motif_concept_set:
                continue
            fs = float(fscore)
            extra_concepts.append((concept, str(trainer_name), fs))
            concept_rows.append(
                {
                    "concept": concept,
                    "type": "non-motif",
                    "rank": None,
                    "source": str(trainer_name),
                    "fscore": fs,
                }
            )

    def _concept_sort_key(r: dict[str, Any]) -> tuple:
        if r.get("type") == "motif":
            return (0, r.get("rank") or 10**9)
        return (1, -float(r.get("fscore") or -1), r.get("concept", ""))

    concept_rows.sort(key=_concept_sort_key)

    # -----------------------------------------------------------------------------
    # 3) Select concepts for heatmaps
    # -----------------------------------------------------------------------------
    # No thresholding/selection logic beyond taking the top N motifs by ranking.
    top_motif_n = max(0, int(top_motif_concepts))
    selected_motif_concepts: list[str] = ranked_motif_concepts[:top_motif_n]

    # Include ALL non-motif concepts (disambiguate duplicates by appending source).
    selected_extra_concepts: list[tuple[str, str, float]] = extra_concepts[:]

    # -----------------------------------------------------------------------------
    # 4) Build synthetic CavTrainers + heatmap data (motif / non-motif / merged)
    # -----------------------------------------------------------------------------
    heatmap_png_paths = {
        "motif": assets_dir / "cavs_similarity_heatmap__motif.png",
        "non-motif": assets_dir / "cavs_similarity_heatmap__non_motif.png",
        "merged": assets_dir / "cavs_similarity_heatmap__merged.png",
    }
    heatmap_data: dict[str, Any] = {"motif": None, "non-motif": None, "merged": None}

    if all_trainers:
        base_trainer = (
            motif_weight_source if motif_weight_source is not None else next(iter(all_trainers.values()))
        )
        try:
            from .cavs import CavTrainer

            def _new_trainer() -> Any:
                t = CavTrainer(
                    getattr(base_trainer, "tpcav"),
                    penalty=getattr(base_trainer, "penalty", "l2"),
                )
                t.cav_weights = {}
                t.cav_fscores = {}
                t.cavs_list = []
                return t

            motif_trainer = _new_trainer()
            extra_trainer = _new_trainer()
            merged_trainer = _new_trainer()

            # fill motif trainer
            if motif_weight_source is not None:
                for c in selected_motif_concepts:
                    if c not in getattr(motif_weight_source, "cav_weights", {}):
                        continue
                    w = motif_weight_source.cav_weights[c]
                    motif_trainer.cav_weights[c] = w
                    motif_trainer.cav_fscores[c] = float(
                        getattr(motif_weight_source, "cav_fscores", {}).get(c, 0.0)
                    )
                    motif_trainer.cavs_list.append(w)

            # fill non-motif trainer (disambiguate duplicates)
            label_counts: dict[str, int] = {}
            for concept, source_name, fs in selected_extra_concepts:
                src_trainer = extra_cav_trainers.get(source_name)
                if src_trainer is None:
                    continue
                if concept not in getattr(src_trainer, "cav_weights", {}):
                    continue
                label = concept
                label_counts[label] = label_counts.get(label, 0) + 1
                if label_counts[label] > 1:
                    label = f"{concept} ({source_name})"
                w = src_trainer.cav_weights[concept]
                extra_trainer.cav_weights[label] = w
                extra_trainer.cav_fscores[label] = float(fs)
                extra_trainer.cavs_list.append(w)

            # merged = motif + non-motif
            for c, w in motif_trainer.cav_weights.items():
                merged_trainer.cav_weights[c] = w
                merged_trainer.cav_fscores[c] = float(motif_trainer.cav_fscores.get(c, 0.0))
                merged_trainer.cavs_list.append(w)
            for c, w in extra_trainer.cav_weights.items():
                merged_trainer.cav_weights[c] = w
                merged_trainer.cav_fscores[c] = float(extra_trainer.cav_fscores.get(c, 0.0))
                merged_trainer.cavs_list.append(w)

            motif_meme_file = None
            if motif_file is not None and motif_file_fmt == "meme":
                motif_meme_file = str(motif_file)

            heatmap_data["motif"] = motif_trainer.plot_cavs_similaritiy_heatmap(
                attributions=attributions,
                concept_list=list(motif_trainer.cav_weights.keys()),
                fscore_thresh=fscore_thresh,
                motif_meme_file=motif_meme_file,
                output_path=str(heatmap_png_paths["motif"]),
            )
            heatmap_data["non-motif"] = extra_trainer.plot_cavs_similaritiy_heatmap(
                attributions=attributions,
                concept_list=list(extra_trainer.cav_weights.keys()),
                fscore_thresh=fscore_thresh,
                motif_meme_file=None,
                output_path=str(heatmap_png_paths["non-motif"]),
            )
            heatmap_data["merged"] = merged_trainer.plot_cavs_similaritiy_heatmap(
                attributions=attributions,
                concept_list=list(merged_trainer.cav_weights.keys()),
                fscore_thresh=fscore_thresh,
                motif_meme_file=motif_meme_file,
                output_path=str(heatmap_png_paths["merged"]),
            )
        except Exception:
            heatmap_data = {"motif": None, "non-motif": None, "merged": None}

    # -----------------------------------------------------------------------------
    # 5) Build JS payload (used by Plotly)
    # -----------------------------------------------------------------------------
    motif_logo_concepts = selected_motif_concepts[:]
    js_payload: dict[str, Any] = {
        "motif_file_fmt": motif_file_fmt,
        "motif_auc_rows": motif_auc_df.to_dict(orient="records") if motif_auc_df is not None else None,
        "motif_insertions": motif_insertions,
        "concept_rows": concept_rows,
        "motif_logos": _maybe_build_motif_logo_data_uris(
            motif_file if motif_file_fmt == "meme" else None,
            motif_logo_concepts,
        ),
        "heatmaps": {
            "motif": {"heatmap": None, "heatmap_div_id": "heatmap__motif", "hover_div_id": "hover__motif"},
            "non-motif": {"heatmap": None, "heatmap_div_id": "heatmap__non_motif", "hover_div_id": "hover__non_motif"},
            "merged": {"heatmap": None, "heatmap_div_id": "heatmap__merged", "hover_div_id": "hover__merged"},
        },
    }

    def _to_list(x: Any) -> Any:
        try:
            return x.tolist()
        except Exception:
            return x

    for key in ("motif", "non-motif", "merged"):
        if isinstance(heatmap_data.get(key), dict):
            js_payload["heatmaps"][key]["heatmap"] = {
                "concept_names_sorted_rows": heatmap_data[key].get("concept_names_sorted_rows"),
                "concept_names_sorted_cols": heatmap_data[key].get("concept_names_sorted_cols"),
                "matrix_similarity_sorted": _to_list(heatmap_data[key].get("matrix_similarity_sorted")),
                "log_ratios_by_attr": heatmap_data[key].get("log_ratios_by_attr"),
            }

    # expose png paths for optional embedding/debug
    js_payload["heatmap_pngs"] = {k: str(p) for k, p in heatmap_png_paths.items()}

    # -----------------------------------------------------------------------------
    # 6) Per-trainer details (collapsed section)
    # -----------------------------------------------------------------------------
    trainer_sections: list[str] = []
    for name, trainer in all_trainers.items():
        meta_rows = [
            ("name", str(name)),
            ("penalty", str(getattr(trainer, "penalty", ""))),
            ("num_cavs", str(len(getattr(trainer, "cav_weights", {}) or {}))),
        ]

        table_html = ""
        cav_fscores = getattr(trainer, "cav_fscores", None)
        if isinstance(cav_fscores, dict) and cav_fscores:
            try:
                import pandas as pd

                df = pd.DataFrame(
                    [{"concept": k, "fscore": float(v)} for k, v in cav_fscores.items()]
                ).sort_values("fscore", ascending=False)
                table_html = _render_df_table(df, max_rows=200)
            except Exception:
                pairs = sorted(cav_fscores.items(), key=lambda kv: kv[1], reverse=True)
                preview = "\n".join(f"{k}\t{v}" for k, v in pairs[:200])
                table_html = f"<pre>{_html.escape(preview)}</pre>"

        trainer_sections.append(
            "\n".join(
                [
                    f"<div class='trainer'><h3>{_html.escape(str(name))}</h3>",
                    _render_kv_table(meta_rows),
                    "<details class='details'>",
                    "<summary class='muted'>CAV F-scores</summary>",
                    table_html or "<div class='muted'>(no cav_fscores found)</div>",
                    "</details>",
                    "</div>",
                ]
            )
        )

    if motif_auc_plot_exists:
        if embed_images:
            motif_src = _read_as_data_uri(motif_auc_plot_path)
        else:
            motif_src = _html.escape(motif_auc_plot_path.name)
        motif_auc_fig_html = f"<img class='figure' src='{motif_src}' alt='Motif AUC regression'/>"
    elif motif_cav_trainers:
        motif_auc_fig_html = "<div class='warn'>Motif AUC regression plot not generated.</div>"
    else:
        motif_auc_fig_html = ""

    motif_auc_table_html = ""
    if motif_auc_df is not None:
        motif_auc_table_html = _render_df_table(motif_auc_df, max_rows=5000)

    if embed_images:
        assets_note = (
            "<div class='muted'>Images are embedded; the assets folder is optional.</div>"
        )
    else:
        assets_note = (
            f"<div class='muted'>Images are stored in: {_html.escape(str(assets_dir))}</div>"
        )

    param_rows = [
        ("fscore_thresh", str(fscore_thresh)),
        ("top_motif_concepts", str(top_motif_concepts)),
        ("motif_file", str(motif_file) if motif_file is not None else ""),
        ("motif_file_fmt", str(motif_file_fmt)),
    ]
    params_html = _render_kv_table(param_rows)

    all_concepts: list[str] = []
    seen = set()
    for c in ranked_motif_concepts:
        if c not in seen:
            all_concepts.append(c)
            seen.add(c)
    for trainer in all_trainers.values():
        cav_fscores = getattr(trainer, "cav_fscores", None)
        if isinstance(cav_fscores, dict):
            for c in cav_fscores.keys():
                c = str(c)
                if c not in seen:
                    all_concepts.append(c)
                    seen.add(c)

    concepts_html = "".join(
        f"<li class='concept'>{_html.escape(c)}</li>" for c in all_concepts[:300]
    )

    motif_concept_rows = [r for r in concept_rows if r.get("type") == "motif"]
    non_motif_concept_rows = [r for r in concept_rows if r.get("type") == "non-motif"]

    motif_table_html = ""
    extra_table_html = ""
    try:
        import pandas as pd

        motif_cols = (
            ["rank", "concept"]
            + [f"fscore_{nm}" for nm in motif_insertions]
            + ["AUC_fscores", "AUC_fscores_residual", "source"]
        )
        extra_cols = ["concept", "source", "fscore"]

        if motif_concept_rows:
            motif_df = pd.DataFrame(motif_concept_rows)
            motif_df = motif_df[[c for c in motif_cols if c in motif_df.columns]]
            motif_table_html = _render_df_table(motif_df, max_rows=5000)
        if non_motif_concept_rows:
            extra_df = pd.DataFrame(non_motif_concept_rows)
            extra_df = extra_df[[c for c in extra_cols if c in extra_df.columns]]
            extra_table_html = _render_df_table(extra_df, max_rows=5000)
    except Exception:
        pass

    payload_json = _json_for_html(js_payload)
    def _barplot_divs(prefix: str, data: Any) -> str:
        if not isinstance(data, dict) or not data.get("log_ratios_by_attr"):
            return ""
        out = ""
        for attr_key in sorted(data["log_ratios_by_attr"].keys()):
            out += f"<div id='bar__{_html.escape(prefix)}__{_html.escape(str(attr_key))}' class='plot plot--short'></div>"
        return out

    def _png_html(path: Path, alt: str) -> str:
        if not embed_images or not path.exists():
            return ""
        return (
            "<img class='figure figure--half' src='"
            + _read_as_data_uri(path)
            + f"' alt='{_html.escape(alt)}'/>"
        )

    motif_barplots_html = _barplot_divs("motif", heatmap_data.get("motif"))
    extra_barplots_html = _barplot_divs("non-motif", heatmap_data.get("non-motif"))
    merged_barplots_html = _barplot_divs("merged", heatmap_data.get("merged"))

    motif_png_html = _png_html(heatmap_png_paths["motif"], "Motif heatmap PNG")
    extra_png_html = _png_html(heatmap_png_paths["non-motif"], "Non-motif heatmap PNG")
    merged_png_html = _png_html(heatmap_png_paths["merged"], "Merged heatmap PNG")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.2/dist/svg2pdf.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #111936;
      --text: #e7ecff;
      --muted: #b7c0e1;
      --warn: #f6c177;
      --err: #ff7a90;
      --border: rgba(231, 236, 255, 0.14);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 18px 60px;
    }}
    header {{
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(17,25,54,0.9), rgba(17,25,54,0.6));
      border-radius: 12px;
      padding: 18px 18px 14px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      letter-spacing: 0.2px;
    }}
    section {{
      margin-top: 18px;
      padding: 14px 14px 18px;
      border: 1px solid var(--border);
      background: rgba(17,25,54,0.65);
      border-radius: 12px;
    }}
    .trainer {{
      margin-top: 14px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }}
    .details summary {{
      cursor: pointer;
      margin-top: 10px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    h3 {{
      margin: 14px 0 10px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.2px;
      text-transform: uppercase;
    }}
    .meta {{
      font-family: var(--mono);
      color: var(--muted);
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .figure {{
      display: block;
      max-width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      margin: 12px 0 6px;
    }}
    .figure--half {{
      max-width: 50%;
    }}
    .plot {{
      width: 100%;
      min-height: 520px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      margin: 12px 0 6px;
    }}
    .plot--short {{
      min-height: 320px;
    }}
    .dlbar {{
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      margin-top: 10px;
    }}
    .btn {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 12px;
    }}
    .btn:hover {{
      background: rgba(255,255,255,0.08);
    }}
    .btn:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}
    .hoverbox {{
      padding: 8px 10px;
      border: 1px dashed var(--border);
      border-radius: 10px;
      margin: 10px 0 0;
      min-height: 20px;
    }}
    .hoverbox img {{
      height: 42px;
      vertical-align: middle;
      margin-left: 8px;
    }}
    .concepts {{
      columns: 3;
      column-gap: 18px;
      padding-left: 18px;
      margin: 10px 0 0;
    }}
    .concept {{
      break-inside: avoid;
      margin: 2px 0;
      font-size: 13px;
      color: var(--text);
    }}
    .table-scroll {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-top: 10px;
      background: rgba(0,0,0,0.12);
    }}
    .table-scroll table.df {{
      margin-top: 0;
    }}
    table.df thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .warn {{
      color: var(--warn);
      font-size: 13px;
      margin-top: 10px;
    }}
    .err {{
      color: var(--err);
      font-family: var(--mono);
      font-size: 12px;
      white-space: pre-wrap;
      margin-top: 8px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }}
    table.df {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      overflow: hidden;
      border-radius: 10px;
      border: 1px solid var(--border);
      font-size: 12px;
    }}
    table.df th, table.df td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    table.df th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
      background: rgba(255,255,255,0.03);
    }}
    table.kv {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 6px;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      font-size: 12px;
    }}
    table.kv th, table.kv td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
    }}
    table.kv th {{
      width: 180px;
      text-align: left;
      color: var(--muted);
      background: rgba(255,255,255,0.03);
      font-weight: 600;
    }}
    pre {{
      font-family: var(--mono);
      font-size: 12px;
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      overflow: auto;
      color: var(--text);
      margin: 10px 0 0;
    }}
  </style>
</head>
<body>
		  <div class="container">
		    <header>
		      <h1>{_html.escape(title)}</h1>
		      <div class="meta">Generated: {now}</div>
		      {params_html}
		      {assets_note}
		    </header>

			    <section>
			      <h2>Concepts</h2>
			      <div class="muted">Motif concepts are listed first (ranked), then any additional concepts.</div>
			      <h3>Motif Concepts</h3>
			      <div class="table-scroll">{motif_table_html or "<div class='muted'>(no motif concepts)</div>"}</div>
		      <h3>Non-motif Concepts</h3>
		      <div class="table-scroll">{extra_table_html or "<div class='muted'>(no non-motif concepts)</div>"}</div>
			      <ol class="concepts">
			        {concepts_html}
			      </ol>
			    </section>
	
			    <section>
			      <h2>Motif Ranking</h2>
			      <div id="motif_auc_plots" class="plot plot--short"></div>
			      {motif_auc_fig_html}
			      <div class="table-scroll">{motif_auc_table_html or "<div class='muted'>(no motif ranking table)</div>"}</div>
			    </section>
	
		    <section>
		      <h2>Trainer Heatmaps</h2>
		      <div class="muted">Motif heatmap (top motifs), non-motif heatmap (all non-motif concepts), and merged heatmap.</div>
		      <h3>Motif</h3>
		      <div id="heatmap__motif" class="plot"></div>
		      <div id="hover__motif" class="hoverbox muted"></div>
		      {motif_barplots_html}
		      {motif_png_html}

		      <h3>Non-motif</h3>
		      <div id="heatmap__non_motif" class="plot"></div>
		      <div id="hover__non_motif" class="hoverbox muted"></div>
		      {extra_barplots_html}
		      {extra_png_html}

		      <h3>Merged</h3>
		      <div id="heatmap__merged" class="plot"></div>
		      <div id="hover__merged" class="hoverbox muted"></div>
		      {merged_barplots_html}
		      {merged_png_html}

		      <details class="details">
		        <summary class="muted">Per-trainer details</summary>
		        {"".join(trainer_sections)}
		      </details>
		    </section>
		  </div>
		  <script type="application/json" id="tpcav-report-data">{payload_json}</script>
		  <script>
		    (function() {{
		      const node = document.getElementById("tpcav-report-data");
		      if (!node) return;
		      const payload = JSON.parse(node.textContent);

          function _ensureDownloadBar(divId, filenameBase) {{
            const div = document.getElementById(divId);
            if (!div || !window.Plotly) return;
            const barId = "dlbar__" + divId;
            if (document.getElementById(barId)) return;

            const bar = document.createElement("div");
            bar.className = "dlbar";
            bar.id = barId;

            const btnSvg = document.createElement("button");
            btnSvg.className = "btn";
            btnSvg.textContent = "Download SVG";

            const btnPdf = document.createElement("button");
            btnPdf.className = "btn";
            btnPdf.textContent = "Download PDF";

            bar.appendChild(btnSvg);
            bar.appendChild(btnPdf);
            div.parentNode.insertBefore(bar, div);

            btnSvg.addEventListener("click", () => {{
              try {{
                Plotly.downloadImage(div, {{format: "svg", filename: filenameBase}});
              }} catch (e) {{
                console.error(e);
              }}
            }});

            function dataUrlToText(dataUrl) {{
              const comma = dataUrl.indexOf(",");
              const meta = dataUrl.slice(0, comma);
              const data = dataUrl.slice(comma + 1);
              if (meta.includes(";base64")) {{
                return atob(data);
              }}
              return decodeURIComponent(data);
            }}

            function lengthToPx(value) {{
              if (!value) return null;
              const v = String(value).trim();
              const n = parseFloat(v);
              if (!Number.isFinite(n)) return null;
              if (v.endsWith("pt")) return n * (96 / 72);
              if (v.endsWith("in")) return n * 96;
              if (v.endsWith("cm")) return n * (96 / 2.54);
              if (v.endsWith("mm")) return n * (96 / 25.4);
              // px or unitless -> treat as px
              return n;
            }}

            function getSvgSizePx(svgEl) {{
              const vb = svgEl.getAttribute("viewBox");
              if (vb) {{
                const parts = vb.split(/\\s+/).map(parseFloat);
                if (parts.length === 4 && parts.every(Number.isFinite)) {{
                  return {{ widthPx: parts[2], heightPx: parts[3] }};
                }}
              }}
              const w = lengthToPx(svgEl.getAttribute("width"));
              const h = lengthToPx(svgEl.getAttribute("height"));
              if (w && h) return {{ widthPx: w, heightPx: h }};
              return null;
            }}

            async function downloadPdf() {{
              // Use raster PDF export for reliability (svg->pdf can clip Plotly heatmaps).
              try {{
                if (!window.jspdf) {{
                  throw new Error("Missing jspdf (CDN blocked?)");
                }}
                const pngUrl = await Plotly.toImage(div, {{format: "png", scale: 3}});
                const img = new Image();
                const loaded = new Promise((resolve, reject) => {{
                  img.onload = resolve;
                  img.onerror = reject;
                }});
                img.src = pngUrl;
                await loaded;

                const {{ jsPDF }} = window.jspdf;
                const orientation = img.naturalWidth >= img.naturalHeight ? "l" : "p";
                const pdf = new jsPDF({{ unit: "pt", format: "a4", orientation }});
                const pageW = pdf.internal.pageSize.getWidth();
                const pageH = pdf.internal.pageSize.getHeight();
                const margin = 18;
                const maxW = pageW - margin * 2;
                const maxH = pageH - margin * 2;
                const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
                const w = img.naturalWidth * scale;
                const h = img.naturalHeight * scale;
                const x = (pageW - w) / 2;
                const y = (pageH - h) / 2;
                pdf.addImage(pngUrl, "PNG", x, y, w, h);
                pdf.save(filenameBase + ".pdf");
              }} catch (e) {{
                console.error(e);
              }}
            }}

            btnPdf.addEventListener("click", () => {{
              downloadPdf();
            }});
          }}

	      function linreg(xs, ys) {{
	        const n = xs.length;
	        if (n === 0) return null;
	        const xmean = xs.reduce((a,b)=>a+b,0) / n;
	        const ymean = ys.reduce((a,b)=>a+b,0) / n;
	        let num = 0, den = 0;
	        for (let i=0; i<n; i++) {{
	          const dx = xs[i] - xmean;
	          num += dx * (ys[i] - ymean);
	          den += dx * dx;
	        }}
	        const slope = den === 0 ? 0 : num / den;
	        const intercept = ymean - slope * xmean;
	        let ssTot = 0, ssRes = 0;
	        for (let i=0; i<n; i++) {{
	          const yhat = slope * xs[i] + intercept;
	          ssTot += (ys[i] - ymean) ** 2;
	          ssRes += (ys[i] - yhat) ** 2;
	        }}
	        const r2 = ssTot === 0 ? 0 : 1 - ssRes/ssTot;
	        return {{slope, intercept, r2}};
	      }}

		      function renderMotifAuc() {{
	        const rows = payload.motif_auc_rows;
	        if (!rows || rows.length === 0) {{
	          return;
	        }}
	        const candidates = [
	          ["information_content_GC", "AUC_fscores"],
	          ["information_content", "AUC_fscores"],
	          ["motif_len", "AUC_fscores"],
	          ["avg_gc", "AUC_fscores"],
	          ["avg_len", "AUC_fscores"],
	        ];
	        const available = candidates.filter(([x,y]) => rows[0][x] !== undefined && rows[0][y] !== undefined).slice(0, 3);
	        if (available.length === 0) return;

	        const traces = [];
	        const layout = {{
	          grid: {{rows: 1, columns: available.length, pattern: "independent"}},
	          paper_bgcolor: "rgba(0,0,0,0)",
	          plot_bgcolor: "rgba(0,0,0,0)",
	          font: {{color: "#e7ecff"}},
	          margin: {{l: 50, r: 20, t: 40, b: 60}},
	          title: "Motif AUC regression (JS)",
	        }};

	        available.forEach(([xcol, ycol], idx) => {{
	          const xs = [];
	          const ys = [];
	          rows.forEach((r) => {{
	            const x = +r[xcol];
	            const y = +r[ycol];
	            if (Number.isFinite(x) && Number.isFinite(y)) {{
	              xs.push(x);
	              ys.push(y);
	            }}
	          }});
	          const fit = linreg(xs, ys);
	          const xMin = Math.min(...xs), xMax = Math.max(...xs);
	          const lineX = [xMin, xMax];
	          const lineY = fit ? [fit.slope*xMin + fit.intercept, fit.slope*xMax + fit.intercept] : [];
	          const axisSuffix = idx === 0 ? "" : String(idx+1);

	          traces.push({{
	            type: "scatter",
	            mode: "markers",
	            x: xs,
	            y: ys,
	            marker: {{color: "rgba(120, 170, 255, 0.85)", size: 7}},
	            xaxis: "x" + axisSuffix,
	            yaxis: "y" + axisSuffix,
	            showlegend: false,
	          }});
	          if (fit) {{
	            traces.push({{
	              type: "scatter",
	              mode: "lines",
	              x: lineX,
	              y: lineY,
	              line: {{color: "rgba(246, 193, 119, 0.95)", width: 2}},
	              xaxis: "x" + axisSuffix,
	              yaxis: "y" + axisSuffix,
	              showlegend: false,
		            }});
		            layout["xaxis" + axisSuffix] = {{title: xcol}};
		            layout["yaxis" + axisSuffix] = {{title: ycol, range: [0, 1]}};
		            layout["annotations"] = (layout["annotations"] || []).concat([{{
		              xref: "x" + axisSuffix + " domain",
		              yref: "y" + axisSuffix + " domain",
	              x: 0.02,
	              y: 0.98,
	              xanchor: "left",
	              yanchor: "top",
	              text: "R²=" + fit.r2.toFixed(4),
	              showarrow: false,
	              font: {{color: "#b7c0e1", size: 12}},
	            }}]);
	          }}
	        }});
		        Plotly.newPlot("motif_auc_plots", traces, layout, {{displayModeBar: false}}).then(() => {{
              _ensureDownloadBar("motif_auc_plots", "motif_auc_plots");
            }});
		      }}

		      function renderHeatmap(kind) {{
		        const hs = (payload.heatmaps || {{}});
		        const entry = hs[kind] || {{}};
		        const h = entry.heatmap || {{}};
		        const z = h.matrix_similarity_sorted;
		        const x = h.concept_names_sorted_cols;
		        const y = h.concept_names_sorted_rows;
		        if (!z || !x || !y) return;

		        const logos = payload.motif_logos || {{}};
		        const images = [];
		        for (let i = 0; i < x.length; i++) {{
		          const name = x[i];
		          const src = logos[name];
		          if (!src) continue;
		          images.push({{
		            source: src,
		            xref: "x",
		            yref: "paper",
		            x: name,
		            y: 1.02,
		            sizex: 0.9,
		            sizey: 0.10,
		            xanchor: "center",
		            yanchor: "bottom",
		            sizing: "contain",
		            layer: "above",
		            opacity: 1.0,
		          }});
		        }}
		        for (let i = 0; i < y.length; i++) {{
		          const name = y[i];
		          const src = logos[name];
		          if (!src) continue;
		          images.push({{
		            source: src,
		            xref: "paper",
		            yref: "y",
		            x: 1.02,
		            y: name,
		            sizex: 0.12,
		            sizey: 0.9,
		            xanchor: "left",
		            yanchor: "middle",
		            sizing: "contain",
		            layer: "above",
		            opacity: 1.0,
		          }});
		        }}

		        Plotly.newPlot(
		          entry.heatmap_div_id,
		          [{{
		            type: "heatmap",
		            z: z,
		            x: x,
		            y: y,
		            zmin: -1,
		            zmax: 1,
		            colorscale: "RdBu",
                    colorbar: {{orientation: 'v', x: 1.10}},
		          }}],
		          {{
		            title: kind.toUpperCase() + " CAV similarity",
		            paper_bgcolor: "rgba(0,0,0,0)",
		            plot_bgcolor: "rgba(0,0,0,0)",
		            font: {{color: "#e7ecff"}},
		            margin: {{l: 160, r: 170, t: 110, b: 120}},
		            xaxis: {{tickangle: 45, automargin: true}},
		            yaxis: {{automargin: true}},
		            images: images,
		          }},
		          {{displayModeBar: false}}
		        ).then(() => {{
              _ensureDownloadBar(entry.heatmap_div_id, "heatmap_" + kind);
            }});

		        const hoverDiv = document.getElementById(entry.hover_div_id);
		        const plotDiv = document.getElementById(entry.heatmap_div_id);
		        function setHoverHtml(html) {{
		          if (!hoverDiv) return;
		          hoverDiv.innerHTML = html;
		        }}
		        if (plotDiv && hoverDiv) {{
		          plotDiv.on('plotly_hover', (ev) => {{
		            const p = ev.points && ev.points[0];
		            if (!p) return;
		            const xName = p.x;
		            const yName = p.y;
		            const xLogo = logos[xName];
		            const yLogo = logos[yName];
		            let html = "<span>Hover:</span> " + xName + " × " + yName;
		            if (xLogo) html += " <img alt='PWM " + xName + "' src='" + xLogo + "'/>";
		            if (yLogo && yName !== xName) html += " <img alt='PWM " + yName + "' src='" + yLogo + "'/>";
		            setHoverHtml(html);
		          }});
		          plotDiv.on('plotly_unhover', () => setHoverHtml(""));
		        }}

		        const logRatios = h.log_ratios_by_attr;
		        if (!logRatios) return;
		        Object.keys(logRatios).forEach((k) => {{
		          const divId = "bar__" + kind + "__" + k;
		          const vals = logRatios[k];
		          if (!vals) return;
		          const colors = vals.map(v => v > 0 ? "rgba(255, 122, 144, 0.95)" : "rgba(120, 170, 255, 0.85)");
		          Plotly.newPlot(
		            divId,
		            [{{
		              type: "bar",
		              x: x,
		              y: vals,
		              marker: {{color: colors}},
		            }}],
		            {{
		              title: "TPCAV score (attr " + k + ")",
		              paper_bgcolor: "rgba(0,0,0,0)",
		              plot_bgcolor: "rgba(0,0,0,0)",
			              font: {{color: "#e7ecff"}},
			              margin: {{l: 40, r: 20, t: 40, b: 140}},
			              xaxis: {{tickangle: 60, automargin: true}},
		              yaxis: {{title: "Log TPCAV score", range: [-5, 5]}},
			            }},
			            {{displayModeBar: false}}
			          ).then(() => {{
                _ensureDownloadBar(divId, "bar_" + kind + "_attr_" + k);
              }});
		        }});
		      }}

	      renderMotifAuc();
	      renderHeatmap("motif");
	      renderHeatmap("non-motif");
	      renderHeatmap("merged");
		    }})();
		  </script>
		</body>
	</html>
	"""

    output_html_path.write_text(html, encoding="utf-8")
    return output_html_path


# Backward-compatible alias (older name used `tcav` instead of `tpcav`).
def generate_tcav_html_report(*args, **kwargs) -> Path:
    return generate_tpcav_html_report(*args, **kwargs)
