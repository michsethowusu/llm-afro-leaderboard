import os
import re
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_available_recipes(recipes_dir="recipes"):
    recipes = []
    if not os.path.isdir(recipes_dir):
        return recipes
    for f in os.listdir(recipes_dir):
        if f.endswith(".py") and f != "__init__.py":
            recipes.append(f[:-3])
    return recipes


def extract_recipe_name_from_filename(filename, available_recipes):
    name = os.path.splitext(filename)[0]
    for recipe in available_recipes:
        if f"_{recipe}" in name:
            return recipe
    m = re.search(r'_([^_]+)$', name)
    return m.group(1) if m else "unknown_recipe"


def collect_results(input_dir="output"):
    results, source_breakdown = {}, {}
    recipes = get_available_recipes()

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            folder = os.path.basename(root)
            if "-" not in folder:
                continue
            src, tgt = folder.split("-", 1)
            recipe = extract_recipe_name_from_filename(file, recipes)
            try:
                df = pd.read_csv(os.path.join(root, file))
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
            if "similarity_score" not in df.columns:
                continue
            avg = df["similarity_score"].mean() * 100
            lp = f"{src}-{tgt}"
            results.setdefault(lp, {})[recipe] = avg
            if "source" in df.columns:
                source_breakdown.setdefault(lp, {}).setdefault(recipe, {})
                for s, g in df.groupby("source"):
                    source_breakdown[lp][recipe][s] = g["similarity_score"].mean() * 100
    return results, source_breakdown


# ---------------------------------------------------------------------
# Chart utilities  (all horizontal + ascending)
# ---------------------------------------------------------------------
def horizontal_bar(data, title, xlabel, filename, outdir):
    """Simple horizontal bar (ascending)."""
    sorted_data = sorted(data.items(), key=lambda x: x[1])
    labels, vals = zip(*sorted_data)
    colors = px.colors.qualitative.Set3
    if len(labels) > len(colors):
        colors *= (len(labels) // len(colors) + 1)

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker=dict(color=colors[:len(labels)],
                    line=dict(color="rgba(50,50,50,0.8)", width=1)),
        text=[f"{v:.2f}%" for v in vals],
        textposition="outside"
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(title=xlabel, gridcolor="lightgray", gridwidth=1),
        yaxis=dict(title="", categoryorder="array", categoryarray=list(labels)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=150, r=80, t=80, b=60),
        height=max(400, len(labels) * 50 + 100)
    )
    fig.write_html(os.path.join(outdir, f"{filename}.html"))
    fig.write_image(os.path.join(outdir, f"{filename}.png"),
                    width=1200, height=max(400, len(labels) * 50 + 100))
    return fig


def stacked_horizontal(data_dict, title, xlabel, filename, outdir):
    """Stacked horizontal (ascending by total)."""
    if not data_dict:
        return
    all_sources = sorted({s for m in data_dict.values() for s in m})
    totals = {m: sum(v.values()) for m, v in data_dict.items()}
    model_order = [m for m, _ in sorted(totals.items(), key=lambda x: x[1])]

    colors = px.colors.qualitative.Set3
    if len(all_sources) > len(colors):
        colors *= (len(all_sources) // len(colors) + 1)

    fig = go.Figure()
    for i, s in enumerate(all_sources):
        vals = [data_dict.get(m, {}).get(s, 0) for m in model_order]
        fig.add_trace(go.Bar(
            name=s, x=vals, y=model_order, orientation="h",
            marker=dict(color=colors[i % len(colors)]),
            text=[f"{v:.1f}%" if v > 0 else "" for v in vals],
            textposition="inside"
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(title=xlabel, gridcolor="lightgray"),
        yaxis=dict(title="", categoryorder="array", categoryarray=model_order),
        barmode="stack",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=150, r=80, t=80, b=60),
        height=max(400, len(model_order) * 60 + 150),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    fig.write_html(os.path.join(outdir, f"{filename}.html"))
    fig.write_image(os.path.join(outdir, f"{filename}.png"),
                    width=1400, height=max(400, len(model_order) * 60 + 150))
    return fig


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def generate_language_specific_reports(results, breakdown, outdir="reports"):
    for lp, model_results in results.items():
        lang_dir = os.path.join(outdir, lp)
        os.makedirs(lang_dir, exist_ok=True)
        if not model_results:
            continue

        # horizontal per-language bar
        horizontal_bar(model_results,
                       f"Translation Quality for {lp}",
                       "Similarity Score (%)",
                       "performance_comparison",
                       lang_dir)

        # stacked source breakdown
        if lp in breakdown and breakdown[lp]:
            stacked_horizontal(breakdown[lp],
                               f"Translation Quality by Source for {lp}",
                               "Similarity Score (%)",
                               "source_breakdown",
                               lang_dir)

        # CSV + summary (still descending)
        df = (pd.DataFrame([
            {"Model": m,
             "Similarity Score (%)": f"{v:.2f}%",
             "Raw Score": v} for m, v in model_results.items()])
              .sort_values("Raw Score", ascending=False))
        df.to_csv(os.path.join(lang_dir, "detailed_report.csv"), index=False)

        if lp in breakdown and breakdown[lp]:
            srows = []
            for m, srcs in breakdown[lp].items():
                for s, v in srcs.items():
                    srows.append({"Model": m, "Source": s,
                                  "Similarity Score (%)": f"{v:.2f}%",
                                  "Raw Score": v})
            pd.DataFrame(srows).to_csv(os.path.join(lang_dir, "source_breakdown.csv"), index=False)

        summary = {
            "language_pair": lp,
            "timestamp": datetime.now().isoformat(),
            "models": model_results,
            "average_score": np.mean(list(model_results.values())),
            "best_model": max(model_results, key=model_results.get),
            "best_score": max(model_results.values())
        }
        if lp in breakdown:
            summary["source_breakdown"] = breakdown[lp]
        with open(os.path.join(lang_dir, "summary_report.json"), "w") as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(lang_dir, "summary.txt"), "w") as f:
            f.write(f"Translation Benchmark Results for {lp}\n")
            f.write(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n\nModel Performance:\n")
            for m, v in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{m}: {v:.2f}%\n")
            f.write(f"\nBest Model: {summary['best_model']} ({summary['best_score']:.2f}%)\n")
            f.write(f"Average Score: {summary['average_score']:.2f}%\n")


def generate_overall_summary(results, breakdown, outdir="reports"):
    if not results:
        return

    # overall model averages
    model_perf = {}
    for m in {mm for v in results.values() for mm in v}:
        vals = [v[m] for v in results.values() if m in v]
        if vals:
            model_perf[m] = np.mean(vals)

    # overall language averages across models
    lang_perf = {lp: np.mean(list(v.values())) for lp, v in results.items() if v}

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_language_pairs": len(results),
        "total_models": len({mm for v in results.values() for mm in v}),
        "model_performance": model_perf,
        "language_performance": lang_perf,
        "best_overall_model": max(model_perf, key=model_perf.get) if model_perf else "none",
        "best_overall_score": max(model_perf.values()) if model_perf else 0,
    }
    with open(os.path.join(outdir, "overall_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if model_perf:
        horizontal_bar(model_perf,
                       "Overall Model Performance Across All Language Pairs",
                       "Average Similarity Score (%)",
                       "overall_model_performance",
                       outdir)

    if lang_perf:
        horizontal_bar(lang_perf,
                       "Language Translation Performance Across Models (DeepSeek, OpenAI OSS, Llama, Google Gemma)",
                       "Average Accuracy Score (%)",
                       "overall_language_performance",
                       outdir)

    return summary


def generate_report(input_dir="output", output_dir="reports"):
    print("Generating performance reports…")
    os.makedirs(output_dir, exist_ok=True)

    results, breakdown = collect_results(input_dir)
    if not results:
        print("No processed results found.")
        return

    generate_language_specific_reports(results, breakdown, output_dir)
    overall = generate_overall_summary(results, breakdown, output_dir)

    print(f"Reports generated in {output_dir}")
    if overall:
        print(f"Best overall model: {overall['best_overall_model']} "
              f"({overall['best_overall_score']:.2f}%)")
    return results, overall


if __name__ == "__main__":
    generate_report()

