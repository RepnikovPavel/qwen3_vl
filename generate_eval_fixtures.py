from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


TEXT_PROMPT = (
    "Transcribe all visible text in reading order. Preserve line breaks. "
    "Render the table as lines whose cells are separated by ` | `. Return only text."
)
FORMULA_PROMPT = (
    "Transcribe every displayed formula as LaTeX in top-to-bottom order. Return only a JSON "
    'object with schema {"formulas":["latex","..."]}. Do not use Markdown fences.'
)
CHART_PROMPT = (
    "Read the chart and return only JSON with keys title, panels, and facts. Each panel must "
    "contain chart_type, title, x_label, y_label, categories, and series. Each series must "
    "contain name and numeric values. Each fact must contain subject, relation, and object. "
    "Do not use Markdown fences."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(figure: Any, path: Path) -> str:
    figure.savefig(
        path,
        dpi=100,
        facecolor="white",
        metadata={"Software": "qwen3-vl-eval"},
        pil_kwargs={"compress_level": 9},
    )
    plt.close(figure)
    return _sha256(path)


def _text_fixture(path: Path) -> dict[str, Any]:
    title = "Multilingual OCR Validation"
    lines = [
        "English: The quick brown fox jumps over 13 lazy dogs.",
        "Русский: Съешь ещё этих мягких французских булок.",
        "Order № QV-2048 · Date: 2026-07-12 · Total: €52.00",
    ]
    headers = ["Item", "Qty", "Price"]
    rows = [
        ["Sensor-A", "3", "12.50"],
        ["Кабель-B", "2", "7.25"],
        ["Total", "5", "52.00"],
    ]
    expected = "\n".join(
        [title, *lines, " | ".join(headers), *(" | ".join(row) for row in rows)]
    )
    figure = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_axis_off()
    axis.text(0.05, 0.93, title, fontsize=25, weight="bold", va="top")
    for index, line in enumerate(lines):
        axis.text(0.05, 0.82 - index * 0.09, line, fontsize=18, va="top")
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        bbox=(0.05, 0.16, 0.9, 0.34),
        colWidths=(0.58, 0.16, 0.26),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(1.4)
        cell.set_edgecolor("#243447")
        cell.set_facecolor("#e9f0f7" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold")
    return {
        "id": "multilingual_text_table",
        "task": "text",
        "image": path.name,
        "image_sha256": _save_figure(figure, path),
        "prompt": TEXT_PROMPT,
        "ground_truth": {"text": expected},
    }


def _formula_fixture(path: Path) -> dict[str, Any]:
    formulas = [
        r"E=mc^2",
        r"\int_0^\infty e^{-x^2}\,dx=\frac{\sqrt{\pi}}{2}",
        r"\sum_{k=1}^{n}k=\frac{n(n+1)}{2}",
        r"\sigma(z)=\frac{1}{1+e^{-z}}",
    ]
    figure = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_axis_off()
    for index, formula in enumerate(formulas):
        axis.text(0.08, 0.84 - index * 0.21, f"${formula}$", fontsize=30, va="center")
    return {
        "id": "formulas_mathtext",
        "task": "formula",
        "image": path.name,
        "image_sha256": _save_figure(figure, path),
        "prompt": FORMULA_PROMPT,
        "ground_truth": {"formulas": formulas},
    }


def _line_bar_ground_truth() -> dict[str, Any]:
    return {
        "title": "Quarterly operations",
        "panels": [
            {
                "chart_type": "line",
                "title": "Request volume",
                "x_label": "Quarter",
                "y_label": "Requests (k)",
                "categories": ["Q1", "Q2", "Q3", "Q4"],
                "series": [
                    {"name": "Alpha", "values": [12, 15, 14, 18]},
                    {"name": "Beta", "values": [10, 13, 16, 17]},
                ],
            },
            {
                "chart_type": "bar",
                "title": "Median latency",
                "x_label": "Backend",
                "y_label": "Seconds",
                "categories": ["CPU", "GPU", "TP2"],
                "series": [{"name": "Latency", "values": [42, 3.2, 1.9]}],
            },
        ],
        "facts": [
            {"subject": "Alpha", "relation": "maximum_at", "object": "Q4"},
            {"subject": "Beta", "relation": "increases_overall", "object": "Q1 to Q4"},
            {"subject": "TP2", "relation": "lowest_value", "object": "1.9"},
        ],
        "numeric_tolerance": 0.05,
    }


def _line_bar_fixture(path: Path) -> dict[str, Any]:
    truth = _line_bar_ground_truth()
    figure, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=100, constrained_layout=True)
    figure.suptitle(truth["title"], fontsize=22, weight="bold")
    quarters = truth["panels"][0]["categories"]
    for series, marker in zip(truth["panels"][0]["series"], ("o", "s"), strict=True):
        axes[0].plot(
            quarters,
            series["values"],
            marker=marker,
            linewidth=2.5,
            label=series["name"],
        )
    axes[0].set_title(truth["panels"][0]["title"])
    axes[0].set_xlabel(truth["panels"][0]["x_label"])
    axes[0].set_ylabel(truth["panels"][0]["y_label"])
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    backends = truth["panels"][1]["categories"]
    values = truth["panels"][1]["series"][0]["values"]
    bars = axes[1].bar(
        backends, values, color=("#4c78a8", "#f58518", "#54a24b"), label="Latency"
    )
    axes[1].set_title(truth["panels"][1]["title"])
    axes[1].set_xlabel(truth["panels"][1]["x_label"])
    axes[1].set_ylabel(truth["panels"][1]["y_label"])
    axes[1].legend()
    axes[1].bar_label(bars, fmt="%g", padding=3)
    return {
        "id": "line_bar_chart",
        "task": "chart",
        "image": path.name,
        "image_sha256": _save_figure(figure, path),
        "prompt": CHART_PROMPT,
        "ground_truth": truth,
    }


def _scatter_heatmap_ground_truth() -> dict[str, Any]:
    return {
        "title": "Calibration diagnostics",
        "panels": [
            {
                "chart_type": "scatter",
                "title": "Observed vs predicted",
                "x_label": "Observed",
                "y_label": "Predicted",
                "categories": ["1", "2", "3", "4", "5"],
                "series": [{"name": "Samples", "values": [1.1, 1.9, 3.2, 3.9, 5.1]}],
            },
            {
                "chart_type": "heatmap",
                "title": "Attention heatmap",
                "x_label": "Column",
                "y_label": "Row",
                "categories": ["C1", "C2", "C3"],
                "series": [
                    {"name": "R1", "values": [0.12, 0.35, 0.48]},
                    {"name": "R2", "values": [0.28, 0.62, 0.71]},
                    {"name": "R3", "values": [0.41, 0.76, 0.93]},
                ],
            },
        ],
        "facts": [
            {
                "subject": "Samples",
                "relation": "positive_trend",
                "object": "Observed to Predicted",
            },
            {"subject": "Maximum", "relation": "located_at", "object": "R3/C3"},
            {"subject": "Maximum", "relation": "value", "object": "0.93"},
        ],
        "numeric_tolerance": 0.05,
    }


def _scatter_heatmap_fixture(path: Path) -> dict[str, Any]:
    truth = _scatter_heatmap_ground_truth()
    figure, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=100, constrained_layout=True)
    figure.suptitle(truth["title"], fontsize=22, weight="bold")
    observed = [1, 2, 3, 4, 5]
    predicted = truth["panels"][0]["series"][0]["values"]
    axes[0].scatter(observed, predicted, s=90, color="#4c78a8", label="Samples")
    axes[0].plot([0.8, 5.2], [0.8, 5.2], linestyle="--", color="#e45756", label="Ideal")
    axes[0].set_title(truth["panels"][0]["title"])
    axes[0].set_xlabel(truth["panels"][0]["x_label"])
    axes[0].set_ylabel(truth["panels"][0]["y_label"])
    axes[0].set_xticks(observed)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    heatmap = [series["values"] for series in truth["panels"][1]["series"]]
    image = axes[1].imshow(heatmap, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(truth["panels"][1]["title"])
    axes[1].set_xlabel(truth["panels"][1]["x_label"])
    axes[1].set_ylabel(truth["panels"][1]["y_label"])
    axes[1].set_xticks(range(3), truth["panels"][1]["categories"])
    axes[1].set_yticks(
        range(3), [series["name"] for series in truth["panels"][1]["series"]]
    )
    for row_index, row in enumerate(heatmap):
        for column_index, value in enumerate(row):
            color = "white" if value < 0.55 else "black"
            axes[1].text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    figure.colorbar(image, ax=axes[1], shrink=0.72)
    return {
        "id": "scatter_heatmap_chart",
        "task": "chart",
        "image": path.name,
        "image_sha256": _save_figure(figure, path),
        "prompt": CHART_PROMPT,
        "ground_truth": truth,
    }


def generate_fixtures(output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fixtures = [
        _text_fixture(destination / "multilingual_text_table.png"),
        _formula_fixture(destination / "formulas_mathtext.png"),
        _line_bar_fixture(destination / "line_bar_chart.png"),
        _scatter_heatmap_fixture(destination / "scatter_heatmap_chart.png"),
    ]
    manifest = {"schema_version": 1, "fixtures": fixtures}
    manifest_path = destination / "manifest.json"
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    args = parser.parse_args()
    manifest = generate_fixtures(args.output_dir)
    print(
        json.dumps({"manifest": str(manifest), "fixture_count": 4}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
