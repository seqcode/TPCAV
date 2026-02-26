from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import mimetypes
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Union


def _sanitize_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "trainer"


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


def generate_tcav_html_report(
    output_html_path: Union[str, Path],
    motif_cav_trainers: Optional[Mapping[Union[int, str], Any]] = None,
    extra_cav_trainers: Optional[Mapping[str, Any]] = None,
    motif_file: Optional[Union[str, Path]] = None,
    motif_file_fmt: str = "meme",
    fscore_thresh: float = 0.8,
    title: str = "TPCAV report",
    embed_images: bool = True,
) -> Path:
    """
    Generate a standalone HTML report from CavTrainer objects.

    Args:
        output_html_path: Path to write the HTML report.
        motif_cav_trainers: Dict mapping "# motif insertions" -> CavTrainer.
        extra_cav_trainers: Optional dict mapping a display name -> CavTrainer.
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
            trainers = [motif_cav_trainers[k] for k in sorted_keys]
            motif_auc_df = compute_motif_auc_fscore(
                num_motif_insertions=[int(k) for k in sorted_keys],
                cav_trainers=trainers,
                motif_file=str(motif_file) if motif_file is not None else None,
                motif_file_fmt=motif_file_fmt,
                output_path=str(motif_auc_plot_path),
            )
            motif_auc_plot_exists = motif_auc_plot_path.exists()
        finally:
            if plt is not None:
                plt.close("all")

    trainer_sections: list[str] = []
    all_trainers: dict[str, Any] = {}
    for k, v in motif_cav_trainers.items():
        all_trainers[f"motifs_{k}"] = v
    all_trainers.update(extra_cav_trainers)

    for name, trainer in all_trainers.items():
        slug = _sanitize_slug(str(name))
        heatmap_path = assets_dir / f"cavs_similarity_heatmap__{slug}.png"

        heatmap_ok = False
        error_msg = None
        try:
            motif_meme_file = None
            if motif_file is not None and motif_file_fmt == "meme":
                motif_meme_file = str(motif_file)

            trainer.plot_cavs_similaritiy_heatmap(
                attributions=None,
                concept_list=None,
                fscore_thresh=fscore_thresh,
                motif_meme_file=motif_meme_file,
                output_path=str(heatmap_path),
            )
            heatmap_ok = heatmap_path.exists()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                pass

        meta_rows = [
            ("name", str(name)),
            ("penalty", str(getattr(trainer, "penalty", ""))),
            ("num_cavs", str(len(getattr(trainer, "cav_weights", {}) or {}))),
            ("fscore_thresh", str(fscore_thresh)),
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

        if heatmap_ok:
            if embed_images:
                src = _read_as_data_uri(heatmap_path)
            else:
                src = _html.escape(heatmap_path.name)
            fig_html = f"<img class='figure figure--half' src='{src}' alt='CAV similarity heatmap: {slug}'/>"
        else:
            fig_html = "<div class='warn'>Heatmap not generated.</div>"
            if error_msg:
                fig_html += f"<pre class='err'>{_html.escape(error_msg)}</pre>"

        trainer_sections.append(
            "\n".join(
                [
                    f"<section><h2>{_html.escape(str(name))}</h2>",
                    _render_kv_table(meta_rows),
                    fig_html,
                    "<h3>CAV F-scores</h3>",
                    table_html or "<div class='muted'>(no cav_fscores found)</div>",
                    "</section>",
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
        motif_auc_table_html = _render_df_table(motif_auc_df, max_rows=200)

    if embed_images:
        assets_note = (
            "<div class='muted'>Images are embedded; the assets folder is optional.</div>"
        )
    else:
        assets_note = (
            f"<div class='muted'>Images are stored in: {_html.escape(str(assets_dir))}</div>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html.escape(title)}</title>
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
      {assets_note}
    </header>

    <section>
      <h2>Motif AUC / regression</h2>
      {motif_auc_fig_html}
      {motif_auc_table_html}
    </section>

    {"".join(trainer_sections)}
  </div>
</body>
</html>
"""

    output_html_path.write_text(html, encoding="utf-8")
    return output_html_path
