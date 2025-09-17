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

# Import language mapping
try:
    from language_mapping import LANGUAGE_MAPPING, get_language_name
except ImportError:
    # Fallback if language_mapping not available
    LANGUAGE_MAPPING = {}
    def get_language_name(lang_code):
        return lang_code

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


def get_language_display_name(language_pair):
    """Convert language pair code to display name using the target language"""
    if '-' not in language_pair:
        return get_language_name(language_pair)
    
    source_lang, target_lang = language_pair.split('-', 1)
    target_name = get_language_name(target_lang)
    
    # If we couldn't get the language name, return the original
    if target_name == target_lang and target_lang in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[target_lang]["name"]
    
    return target_name if target_name != target_lang else language_pair
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


def combine_all_datasets(input_dir="output"):
    """
    Combine all CSV files from all directories into one main dataset
    """
    all_data = []
    recipes = get_available_recipes()
    
    print("Combining all datasets...")
    
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
                if "similarity_score" not in df.columns:
                    continue
                
                # Add metadata columns
                df['language_pair'] = f"{src}-{tgt}"
                df['source_lang'] = src
                df['target_lang'] = tgt
                df['model'] = recipe
                df['file_path'] = os.path.join(root, file)
                
                # Convert similarity score to percentage
                df['similarity_score_pct'] = df['similarity_score'] * 100
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
    if not all_data:
        print("No data files found!")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Language pairs: {combined_df['language_pair'].nunique()}")
    print(f"Models: {combined_df['model'].nunique()}")
    
    return combined_df


def calculate_metrics(df):
    """
    Calculate performance metrics for quadrant analysis
    """
    if df.empty:
        return pd.DataFrame()
    
    # Calculate metrics by model and language pair
    metrics = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Overall metrics
        avg_score = model_data['similarity_score_pct'].mean()
        consistency = 100 - model_data['similarity_score_pct'].std()  # Higher std = lower consistency
        data_points = len(model_data)
        coverage = model_data['language_pair'].nunique()  # Number of language pairs covered
        
        # Performance across language pairs
        lang_performance = model_data.groupby('language_pair')['similarity_score_pct'].mean()
        versatility = 100 - lang_performance.std()  # Lower std across languages = higher versatility
        
        # Source breakdown if available
        if 'source' in model_data.columns:
            source_performance = model_data.groupby('source')['similarity_score_pct'].mean()
            source_consistency = 100 - source_performance.std()
        else:
            source_consistency = consistency
        
        metrics.append({
            'model': model,
            'avg_score': avg_score,
            'consistency': max(0, consistency),  # Ensure non-negative
            'versatility': max(0, versatility),  # Ensure non-negative
            'source_consistency': max(0, source_consistency),
            'coverage': coverage,
            'data_points': data_points,
            'max_score': model_data['similarity_score_pct'].max(),
            'min_score': model_data['similarity_score_pct'].min()
        })
    
    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------
# Quadrant Chart utilities
# ---------------------------------------------------------------------
def create_language_performance_quadrant(metrics_df, x_metric, y_metric, title, filename, outdir, 
                                       size_metric=None, color_metric=None):
    """
    Create a quadrant chart showing language pair positioning (specialized for language data)
    """
    if metrics_df.empty:
        return None
    
    # Calculate quadrant lines (medians)
    x_line = metrics_df[x_metric].median()
    y_line = metrics_df[y_metric].median()
    
    # Determine point sizes
    if size_metric and size_metric in metrics_df.columns:
        sizes = metrics_df[size_metric]
        size_range = [10, 30]
    else:
        sizes = [20] * len(metrics_df)
        size_range = [20, 20]
    
    # Determine colors
    if color_metric and color_metric in metrics_df.columns:
        colors = metrics_df[color_metric]
        color_scale = 'Viridis'
    else:
        colors = px.colors.qualitative.Set3[:len(metrics_df)]
        color_scale = None
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add quadrant background regions
    fig.add_shape(
        type="rect",
        x0=metrics_df[x_metric].min() * 0.95, x1=x_line,
        y0=y_line, y1=metrics_df[y_metric].max() * 1.05,
        fillcolor="lightblue", opacity=0.2, line_width=0
    )
    fig.add_annotation(
        x=x_line * 0.7, y=y_line + (metrics_df[y_metric].max() - y_line) * 0.5,
        text="High Consistency<br>Low Performance", showarrow=False,
        font=dict(size=12, color="gray"), opacity=0.7
    )
    
    fig.add_shape(
        type="rect",
        x0=x_line, x1=metrics_df[x_metric].max() * 1.05,
        y0=y_line, y1=metrics_df[y_metric].max() * 1.05,
        fillcolor="lightgreen", opacity=0.2, line_width=0
    )
    fig.add_annotation(
        x=x_line + (metrics_df[x_metric].max() - x_line) * 0.5,
        y=y_line + (metrics_df[y_metric].max() - y_line) * 0.5,
        text="OPTIMAL LANGUAGES<br>(High Performance & Consistency)", showarrow=False,
        font=dict(size=14, color="darkgreen", weight="bold"), opacity=0.8
    )
    
    fig.add_shape(
        type="rect",
        x0=metrics_df[x_metric].min() * 0.95, x1=x_line,
        y0=metrics_df[y_metric].min() * 0.95, y1=y_line,
        fillcolor="lightcoral", opacity=0.2, line_width=0
    )
    fig.add_annotation(
        x=x_line * 0.7, y=y_line * 0.7,
        text="CHALLENGING LANGUAGES<br>(Low Performance & Consistency)", showarrow=False,
        font=dict(size=12, color="darkred"), opacity=0.7
    )
    
    fig.add_shape(
        type="rect",
        x0=x_line, x1=metrics_df[x_metric].max() * 1.05,
        y0=metrics_df[y_metric].min() * 0.95, y1=y_line,
        fillcolor="lightyellow", opacity=0.2, line_width=0
    )
    fig.add_annotation(
        x=x_line + (metrics_df[x_metric].max() - x_line) * 0.5, y=y_line * 0.7,
        text="High Performance<br>Low Consistency", showarrow=False,
        font=dict(size=12, color="orange"), opacity=0.7
    )
    
    # Add quadrant lines
    fig.add_vline(x=x_line, line_dash="dash", line_color="gray", line_width=2)
    fig.add_hline(y=y_line, line_dash="dash", line_color="gray", line_width=2)
    
    # Add data points
    for i, row in metrics_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[x_metric]],
            y=[row[y_metric]],
            mode='markers+text',
            name=row['language_pair'],
            text=[row['language_pair']],
            textposition="middle right",
            textfont=dict(size=10, color="black"),
            marker=dict(
                size=sizes.iloc[i] if hasattr(sizes, 'iloc') else sizes[i] if hasattr(sizes, '__getitem__') else sizes,
                color=colors.iloc[i] if hasattr(colors, 'iloc') else colors[i] if isinstance(colors, list) else colors,
                colorscale=color_scale if color_scale else None,
                line=dict(width=2, color="white"),
                opacity=0.8
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['language_pair']}</b><br>" +
                f"{x_metric.replace('_', ' ').title()}: {row[x_metric]:.2f}<br>" +
                f"{y_metric.replace('_', ' ').title()}: {row[y_metric]:.2f}<br>" +
                (f"{size_metric.replace('_', ' ').title()}: {row[size_metric]}<br>" if size_metric else "") +
                "<extra></extra>"
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(
            title=x_metric.replace('_', ' ').title(),
            gridcolor="lightgray",
            gridwidth=1,
            showgrid=True
        ),
        yaxis=dict(
            title=y_metric.replace('_', ' ').title(),
            gridcolor="lightgray",
            gridwidth=1,
            showgrid=True
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=120, t=100, b=80),
        height=600,
        width=800
    )
    
    # Save chart
    fig.write_html(os.path.join(outdir, f"{filename}.html"))
    fig.write_image(os.path.join(outdir, f"{filename}.png"), width=800, height=600)
    
    return fig


def create_enhanced_quadrant_chart(metrics_df, x_metric, y_metric, title, filename, outdir, 
                                 id_column, size_metric=None, color_metric=None):
    """
    Create an enhanced quadrant chart with clean label-only visualization
    and quadrant labels positioned only on top and bottom axes
    """
    if metrics_df.empty:
        return None
    
    # Calculate quadrant lines (medians)
    x_line = metrics_df[x_metric].median()
    y_line = metrics_df[y_metric].median()
    
    # Create scatter plot
    fig = go.Figure()
    
    # Calculate axis ranges with some padding
    x_min, x_max = metrics_df[x_metric].min(), metrics_df[x_metric].max()
    y_min, y_max = metrics_df[y_metric].min(), metrics_df[y_metric].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = x_range * 0.05
    y_padding = y_range * 0.05
    
    # Add quadrant background regions with labels positioned on axis lines
    regions = [
        {
            'x0': x_min - x_padding, 'x1': x_line,
            'y0': y_line, 'y1': y_max + y_padding,
            'color': 'lightblue', 
            'label': 'Consistent\nBut Average',
            # Moved to top axis (left half)
            'label_x': x_min + (x_line - x_min) * 0.5,
            'label_y': y_max + y_padding * 0.8,
            'align': 'center'
        },
        {
            'x0': x_line, 'x1': x_max + x_padding,
            'y0': y_line, 'y1': y_max + y_padding,
            'color': 'lightgreen', 
            'label': 'LEADERS\n(High Performance & Consistency)',
            # Position label on the top axis (right half)
            'label_x': x_line + (x_max - x_line) * 0.5,
            'label_y': y_max + y_padding * 0.8,
            'align': 'center'
        },
        {
            'x0': x_min - x_padding, 'x1': x_line,
            'y0': y_min - y_padding, 'y1': y_line,
            'color': 'lightcoral', 
            'label': 'NEEDS IMPROVEMENT\n(Low Performance & Consistency)',
            # Moved to bottom axis (left half)
            'label_x': x_min + (x_line - x_min) * 0.5,
            'label_y': y_min - y_padding * 0.8,
            'align': 'center'
        },
        {
            'x0': x_line, 'x1': x_max + x_padding,
            'y0': y_min - y_padding, 'y1': y_line,
            'color': 'lightyellow', 
            'label': 'High Performance\nBut Inconsistent',
            # Position label on the bottom axis (right half)
            'label_x': x_line + (x_max - x_line) * 0.5,
            'label_y': y_min - y_padding * 0.8,
            'align': 'center'
        }
    ]
    
    for region in regions:
        fig.add_shape(
            type="rect",
            x0=region['x0'], x1=region['x1'],
            y0=region['y0'], y1=region['y1'],
            fillcolor=region['color'], opacity=0.2, line_width=0
        )
        
        # Add region labels positioned on axis lines
        label_color = 'darkgreen' if 'LEADERS' in region['label'] else 'darkred' if 'NEEDS' in region['label'] else 'gray'
        font_weight = 'bold' if 'LEADERS' in region['label'] else 'normal'
        
        fig.add_annotation(
            x=region['label_x'], 
            y=region['label_y'],
            text=region['label'], 
            showarrow=False,
            xanchor=region['align'],
            font=dict(size=12, color=label_color, weight=font_weight), 
            opacity=0.8,
            bgcolor="white",
            bordercolor=label_color,
            borderwidth=1,
            borderpad=4
        )
    
    # Add quadrant lines
    fig.add_vline(x=x_line, line_dash="dash", line_color="gray", line_width=2)
    fig.add_hline(y=y_line, line_dash="dash", line_color="gray", line_width=2)
    
    # Use a force-directed approach to position data labels
    label_positions = {}
    
    # First pass: initial positions
    for i, row in metrics_df.iterrows():
        x_val = row[x_metric]
        y_val = row[y_metric]
        
        # Start with the actual data position (we'll only show labels, not points)
        label_positions[i] = {
            'x': x_val,
            'y': y_val,
            'original_x': x_val,
            'original_y': y_val
        }
    
    # Second pass: resolve overlaps with a simple force-directed algorithm
    max_iterations = 100
    for iteration in range(max_iterations):
        moved = False
        for i, pos_i in label_positions.items():
            for j, pos_j in label_positions.items():
                if i >= j:
                    continue
                
                # Calculate distance between labels
                dx = pos_j['x'] - pos_i['x']
                dy = pos_j['y'] - pos_i['y']
                distance = (dx**2 + dy**2)**0.5
                
                # If labels are too close, push them apart
                min_distance = x_range * 0.08  # Minimum distance between labels
                if distance < min_distance and distance > 0:
                    # Calculate repulsion force
                    force = (min_distance - distance) / distance
                    
                    # Apply force
                    label_positions[i]['x'] -= dx * force * 0.5
                    label_positions[i]['y'] -= dy * force * 0.5
                    label_positions[j]['x'] += dx * force * 0.5
                    label_positions[j]['y'] += dy * force * 0.5
                    
                    moved = True
        
        # Also pull labels toward their original positions
        for i, pos in label_positions.items():
            original_dx = pos['original_x'] - pos['x']
            original_dy = pos['original_y'] - pos['y']
            original_dist = (original_dx**2 + original_dy**2)**0.5
            
            if original_dist > x_range * 0.15:  # If too far from original position
                # Pull back toward original position
                label_positions[i]['x'] += original_dx * 0.1
                label_positions[i]['y'] += original_dy * 0.1
                moved = True
        
        if not moved:
            break
    
    # Add only text labels (no points or connecting lines)
    for i, row in metrics_df.iterrows():
        label_text = row[id_column]
        
        # Get the calculated label position
        label_pos = label_positions[i]
        
        # Add the label
        fig.add_trace(go.Scatter(
            x=[label_pos['x']],
            y=[label_pos['y']],
            mode='text',
            text=[label_text],
            textposition="middle center",
            textfont=dict(
                size=12, 
                color="black",
                family="Arial"
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=(
                f"<b>{label_text}</b><br>" +
                f"{x_metric.replace('_', ' ').title()}: {row[x_metric]:.2f}<br>" +
                f"{y_metric.replace('_', ' ').title()}: {row[y_metric]:.2f}<br>" +
                (f"{size_metric.replace('_', ' ').title()}: {row[size_metric]}<br>" if size_metric else "") +
                (f"{color_metric.replace('_', ' ').title()}: {row[color_metric]:.2f}<br>" if color_metric else "")
            )
        ))
    
    # Adjust layout with extra padding for top and bottom quadrant labels
    chart_height = 900  # Increased height to accommodate top/bottom labels
    chart_width = 1000
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(
            title=x_metric.replace('_', ' ').title(),
            gridcolor="lightgray",
            gridwidth=1,
            showgrid=True,
            range=[x_min - x_padding, x_max + x_padding]
        ),
        yaxis=dict(
            title=y_metric.replace('_', ' ').title(),
            gridcolor="lightgray",
            gridwidth=1,
            showgrid=True,
            range=[y_min - y_padding * 2, y_max + y_padding * 2]  # Extra padding for labels
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100, r=100, t=120, b=120),  # Increased top/bottom margins
        height=chart_height,
        width=chart_width
    )
    
    # Save chart
    fig.write_html(os.path.join(outdir, f"{filename}.html"))
    fig.write_image(os.path.join(outdir, f"{filename}.png"), width=chart_width, height=chart_height)
    
    return fig


def create_language_quadrant(df, title, filename, outdir):
    """
    Create quadrant chart for language pair performance with proper language names
    """
    if df.empty:
        return None
    
    # Calculate metrics by language pair
    lang_metrics = []
    for lp in df['language_pair'].unique():
        lp_data = df[df['language_pair'] == lp]
        
        avg_score = lp_data['similarity_score_pct'].mean()
        consistency = 100 - lp_data['similarity_score_pct'].std()
        model_count = lp_data['model'].nunique()
        
        # Get display name for the language pair
        display_name = get_language_display_name(lp)
        
        lang_metrics.append({
            'language_pair': lp,
            'display_name': display_name,
            'avg_score': avg_score,
            'consistency': max(0, consistency),
            'model_coverage': model_count
        })
    
    lang_df = pd.DataFrame(lang_metrics)
    
    if lang_df.empty:
        return None
    
    return create_enhanced_quadrant_chart(
        lang_df, 'avg_score', 'consistency', title, filename, outdir,
        id_column='display_name', size_metric='model_coverage', color_metric='avg_score'
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def generate_comprehensive_reports(combined_df, outdir="reports"):
    """
    Generate streamlined comprehensive reports with only essential quadrant charts
    """
    if combined_df.empty:
        print("No data available for reporting")
        return
    
    os.makedirs(outdir, exist_ok=True)
    
    # Save combined dataset
    combined_df.to_csv(os.path.join(outdir, "combined_dataset.csv"), index=False)
    print(f"Combined dataset saved with {len(combined_df)} records")
    
    # Calculate metrics for models
    metrics_df = calculate_metrics(combined_df)
    if not metrics_df.empty:
        metrics_df.to_csv(os.path.join(outdir, "model_metrics.csv"), index=False)
    
    # Generate only essential quadrant charts
    
    # 1. Model Performance vs Consistency Quadrant
    create_enhanced_quadrant_chart(
        metrics_df, 'avg_score', 'consistency',
        'Model Performance vs Consistency Analysis',
        'model_performance_quadrant', outdir,
        id_column='model', size_metric='coverage', color_metric='versatility'
    )
    
    # 2. Language Performance Quadrant with proper names
    create_language_quadrant(
        combined_df,
        'Language Performance vs Consistency Analysis',
        'language_performance_quadrant', outdir
    )
    
    # Generate enhanced summary
    generate_enhanced_summary(combined_df, metrics_df, outdir)


def generate_enhanced_summary(combined_df, metrics_df, outdir):
    """
    Generate enhanced summary with insights
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_overview": {
            "total_records": int(len(combined_df)),
            "language_pairs": int(combined_df['language_pair'].nunique()),
            "models": int(combined_df['model'].nunique()),
            "average_score_overall": float(combined_df['similarity_score_pct'].mean()),
            "score_std_overall": float(combined_df['similarity_score_pct'].std())
        }
    }
    
    if not metrics_df.empty:
        # Model insights
        best_performer = metrics_df.loc[metrics_df['avg_score'].idxmax()]
        most_consistent = metrics_df.loc[metrics_df['consistency'].idxmax()]
        most_versatile = metrics_df.loc[metrics_df['versatility'].idxmax()]
        
        # Define leaders (top right quadrant)
        perf_threshold = float(metrics_df['avg_score'].median())
        cons_threshold = float(metrics_df['consistency'].median())
        leaders = metrics_df[
            (metrics_df['avg_score'] >= perf_threshold) & 
            (metrics_df['consistency'] >= cons_threshold)
        ]
        
        summary["model_insights"] = {
            "best_performer": {
                "model": str(best_performer['model']),
                "score": float(best_performer['avg_score'])
            },
            "most_consistent": {
                "model": str(most_consistent['model']),
                "consistency": float(most_consistent['consistency'])
            },
            "most_versatile": {
                "model": str(most_versatile['model']),
                "versatility": float(most_versatile['versatility'])
            },
            "leaders": [str(x) for x in leaders['model'].tolist()],
            "performance_threshold": perf_threshold,
            "consistency_threshold": cons_threshold
        }
    
    # Language pair insights
    lang_performance = combined_df.groupby('language_pair')['similarity_score_pct'].agg(['mean', 'std', 'count'])
    best_language = lang_performance['mean'].idxmax()
    most_challenging = lang_performance['mean'].idxmin()
    
    summary["language_insights"] = {
        "best_performing_pair": {
            "pair": str(best_language),
            "score": float(lang_performance.loc[best_language, 'mean'])
        },
        "most_challenging_pair": {
            "pair": str(most_challenging),
            "score": float(lang_performance.loc[most_challenging, 'mean'])
        },
        "most_evaluated_pair": {
            "pair": str(lang_performance['count'].idxmax()),
            "evaluations": int(lang_performance['count'].max())
        }
    }
    
    # Save summary
    with open(os.path.join(outdir, "enhanced_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate text summary
    with open(os.path.join(outdir, "executive_summary.txt"), "w") as f:
        f.write("TRANSLATION MODEL PERFORMANCE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Records: {summary['dataset_overview']['total_records']:,}\n")
        f.write(f"Language Pairs: {summary['dataset_overview']['language_pairs']}\n")
        f.write(f"Models Evaluated: {summary['dataset_overview']['models']}\n")
        f.write(f"Overall Average Score: {summary['dataset_overview']['average_score_overall']:.2f}%\n\n")
        
        if "model_insights" in summary:
            f.write("KEY FINDINGS\n")
            f.write("-" * 20 + "\n")
            f.write(f"🏆 Best Performer: {summary['model_insights']['best_performer']['model']} ")
            f.write(f"({summary['model_insights']['best_performer']['score']:.2f}%)\n")
            f.write(f"🎯 Most Consistent: {summary['model_insights']['most_consistent']['model']} ")
            f.write(f"({summary['model_insights']['most_consistent']['consistency']:.2f}%)\n")
            f.write(f"🔄 Most Versatile: {summary['model_insights']['most_versatile']['model']} ")
            f.write(f"({summary['model_insights']['most_versatile']['versatility']:.2f}%)\n")
            
            if summary['model_insights']['leaders']:
                f.write(f"👑 Market Leaders: {', '.join(summary['model_insights']['leaders'])}\n\n")
        
        f.write("LANGUAGE INSIGHTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"🌟 Best Language Pair: {summary['language_insights']['best_performing_pair']['pair']} ")
        f.write(f"({summary['language_insights']['best_performing_pair']['score']:.2f}%)\n")
        f.write(f"⚡ Most Challenging: {summary['language_insights']['most_challenging_pair']['pair']} ")
        f.write(f"({summary['language_insights']['most_challenging_pair']['score']:.2f}%)\n")
        f.write(f"📊 Most Evaluated: {summary['language_insights']['most_evaluated_pair']['pair']} ")
        f.write(f"({summary['language_insights']['most_evaluated_pair']['evaluations']} evaluations)\n")
    
    return summary


def generate_report(input_dir="output", output_dir="reports"):
    """
    Main reporting function with combined dataset approach and streamlined quadrant analysis
    """
    print("Generating performance reports with quadrant analysis...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Combine all datasets
    combined_df = combine_all_datasets(input_dir)
    
    if combined_df.empty:
        print("No data found for reporting.")
        return None, None
    
    # Step 2: Generate streamlined reports with only essential charts
    generate_comprehensive_reports(combined_df, output_dir)
    
    # Step 3: Generate language-specific reports (legacy format for compatibility)
    results, breakdown = collect_results_from_combined(combined_df)
    generate_language_specific_reports(results, breakdown, output_dir)
    
    print(f"Reports generated in {output_dir}")
    print("Key outputs:")
    print("- combined_dataset.csv: Full combined dataset")
    print("- model_metrics.csv: Calculated model metrics")  
    print("- model_performance_quadrant.html/png: Model performance analysis")
    print("- language_performance_quadrant.html/png: Language analysis with proper names")
    print("- enhanced_summary.json: Detailed insights")
    print("- executive_summary.txt: Human-readable summary")
    
    return combined_df, results


def collect_results_from_combined(combined_df):
    """
    Extract results and breakdown from combined dataset (for backward compatibility)
    """
    results = {}
    source_breakdown = {}
    
    for lp in combined_df['language_pair'].unique():
        lp_data = combined_df[combined_df['language_pair'] == lp]
        
        # Model results
        model_results = lp_data.groupby('model')['similarity_score_pct'].mean().to_dict()
        results[lp] = model_results
        
        # Source breakdown if available
        if 'source' in combined_df.columns:
            source_breakdown[lp] = {}
            for model in lp_data['model'].unique():
                model_data = lp_data[lp_data['model'] == model]
                if 'source' in model_data.columns:
                    source_results = model_data.groupby('source')['similarity_score_pct'].mean().to_dict()
                    source_breakdown[lp][model] = source_results
    
    return results, source_breakdown


def generate_language_specific_reports(results, breakdown, outdir="reports"):
    """
    Generate individual language reports (kept for backward compatibility)
    """
    for lp, model_results in results.items():
        lang_dir = os.path.join(outdir, "language_pairs", lp)
        os.makedirs(lang_dir, exist_ok=True)
        if not model_results:
            continue

        # CSV + summary
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


if __name__ == "__main__":
    generate_report()
