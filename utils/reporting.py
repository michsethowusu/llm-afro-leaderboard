import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import re

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def collect_results(input_dir="/content/africa-mt-benchmark/output"):
    """Collect all results from processed CSV files in the output directory"""
    results = {}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                folder_name = os.path.basename(root)
                if '-' in folder_name:  # Only process folders with language pairs
                    source_lang, target_lang = folder_name.split('-', 1)
                    
                    # Extract recipe name from filename (pattern: filename_recipe.csv)
                    match = re.search(r'_([^_]+)\.csv$', file)
                    if match:
                        recipe_name = match.group(1)
                        
                        # Read the CSV file
                        df = pd.read_csv(os.path.join(root, file))
                        
                        # Check if the file has been processed (has similarity_score column)
                        if 'similarity_score' in df.columns:
                            avg_score = df['similarity_score'].mean()
                            results.setdefault(f"{source_lang}-{target_lang}", {})[recipe_name] = avg_score * 100  # Convert to percentage
                    
    return results

# The rest of the reporting functions remain the same...
def generate_visualizations(results, output_dir="/content/africa-mt-benchmark/reports"):
    """Generate visualizations from the results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    languages = list(results.keys())
    models = set()
    for lang in languages:
        models.update(results[lang].keys())
    models = list(models)
    
    # Create data for bar chart
    data = []
    for lang in languages:
        for model in models:
            if model in results[lang]:
                data.append({
                    'Language Pair': lang,
                    'Model': model,
                    'Similarity Score (%)': results[lang][model]
                })
    
    df = pd.DataFrame(data)
    
    # Generate bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Language Pair', y='Similarity Score (%)', hue='Model', data=df)
    plt.title('Translation Quality by Language Pair and Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
    plt.close()
    
    # Generate detailed report CSV
    report_data = []
    for lang in languages:
        row = {'Language Pair': lang}
        for model in models:
            if model in results[lang]:
                row[model] = f"{results[lang][model]:.2f}%"
            else:
                row[model] = "N/A"
        report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(os.path.join(output_dir, 'detailed_report.csv'), index=False)
    
    # Generate summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_language_pairs': len(languages),
        'total_models': len(models),
        'overall_average': df['Similarity Score (%)'].mean(),
        'by_language': {lang: np.mean([results[lang].get(model, 0) for model in models]) for lang in languages},
        'by_model': {model: np.mean([results[lang].get(model, 0) for lang in languages]) for model in models}
    }
    
    # Save summary as JSON
    with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate model comparison chart
    if len(models) > 1:
        model_avgs = [summary['by_model'][model] for model in models]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=model_avgs)
        plt.title('Average Performance by Model')
        plt.ylabel('Similarity Score (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
    
    return df, summary

def generate_report(input_dir="/content/africa-mt-benchmark/output", output_dir="/content/africa-mt-benchmark/reports"):
    """Main function to generate reports"""
    print("Generating performance reports...")
    
    # Collect results from all processed files
    results = collect_results(input_dir)
    
    if not results:
        print("No processed results found. Please run translations first.")
        return
    
    # Generate visualizations and reports
    df, summary = generate_visualizations(results, output_dir)
    
    print(f"Reports generated successfully in {output_dir}/")
    print(f"Overall average similarity score: {summary['overall_average']:.2f}%")
    
    return df, summary
