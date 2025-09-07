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

def get_available_recipes(recipes_dir="/content/africa-mt-benchmark/recipes"):
    """Dynamically discover all available recipes"""
    recipes = []
    for file in os.listdir(recipes_dir):
        if file.endswith(".py") and file != "__init__.py":
            recipe_name = file[:-3]  # Remove .py extension
            recipes.append(recipe_name)
    return recipes

def extract_recipe_name_from_filename(filename, available_recipes):
    """Extract recipe name from filename by matching against available recipes"""
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try to find which recipe name is in the filename
    for recipe in available_recipes:
        if f"_{recipe}" in name_without_ext:
            return recipe
    
    # If no match found, try to extract using pattern
    match = re.search(r'_([^_]+)$', name_without_ext)
    if match:
        return match.group(1)
    
    return "unknown_recipe"

def collect_results(input_dir="/content/africa-mt-benchmark/output"):
    """Collect all results from processed CSV files in the output directory"""
    results = {}
    available_recipes = get_available_recipes()
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                folder_name = os.path.basename(root)
                if '-' in folder_name:  # Only process folders with language pairs
                    source_lang, target_lang = folder_name.split('-', 1)
                    
                    # Extract recipe name from filename
                    recipe_name = extract_recipe_name_from_filename(file, available_recipes)
                    
                    # Read the CSV file
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        
                        # Check if the file has been processed (has similarity_score column)
                        if 'similarity_score' in df.columns:
                            avg_score = df['similarity_score'].mean()
                            results.setdefault(f"{source_lang}-{target_lang}", {})[recipe_name] = avg_score * 100  # Convert to percentage
                    except Exception as e:
                        print(f"Error reading {file}: {str(e)}")
                        continue
                    
    return results

def generate_language_specific_reports(results, output_dir="/content/africa-mt-benchmark/reports"):
    """Generate individual reports for each language pair"""
    for language_pair, model_results in results.items():
        # Create language-specific directory
        lang_output_dir = os.path.join(output_dir, language_pair)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        # Prepare data for visualization
        models = list(model_results.keys())
        scores = list(model_results.values())
        
        if not models:
            print(f"No model results found for {language_pair}")
            continue
            
        # Generate language-specific bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores)
        plt.title(f'Translation Quality for {language_pair}')
        plt.ylabel('Similarity Score (%)')
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{score:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(lang_output_dir, 'performance_comparison.png'), dpi=300)
        plt.close()
        
        # Generate language-specific CSV report
        report_data = []
        for model, score in model_results.items():
            report_data.append({
                'Model': model,
                'Similarity Score (%)': f"{score:.2f}%",
                'Raw Score': score
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(os.path.join(lang_output_dir, 'detailed_report.csv'), index=False)
        
        # Generate language-specific summary
        summary = {
            'language_pair': language_pair,
            'timestamp': datetime.now().isoformat(),
            'models': model_results,
            'average_score': np.mean(scores) if scores else 0,
            'best_model': max(model_results, key=model_results.get) if model_results else "none",
            'best_score': max(scores) if scores else 0
        }
        
        # Save summary as JSON
        with open(os.path.join(lang_output_dir, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Generated report for {language_pair} in {lang_output_dir}/")
        
        # Also create a simple text summary
        with open(os.path.join(lang_output_dir, 'summary.txt'), 'w') as f:
            f.write(f"Translation Benchmark Results for {language_pair}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nModel Performance:\n")
            for model, score in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{model}: {score:.2f}%\n")
            f.write(f"\nBest Model: {summary['best_model']} ({summary['best_score']:.2f}%)\n")
            f.write(f"Average Score: {summary['average_score']:.2f}%\n")

def generate_overall_summary(results, output_dir="/content/africa-mt-benchmark/reports"):
    """Generate an overall summary across all language pairs"""
    if not results:
        return
        
    # Prepare data for overall summary
    all_models = set()
    for lang_results in results.values():
        all_models.update(lang_results.keys())
    
    # Calculate average performance per model across all languages
    model_performance = {}
    for model in all_models:
        scores = []
        for lang_results in results.values():
            if model in lang_results:
                scores.append(lang_results[model])
        if scores:
            model_performance[model] = np.mean(scores)
    
    # Create overall summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_language_pairs': len(results),
        'total_models': len(all_models),
        'model_performance': model_performance,
        'best_overall_model': max(model_performance, key=model_performance.get) if model_performance else "none",
        'best_overall_score': max(model_performance.values()) if model_performance else 0,
        'language_pairs': list(results.keys())
    }
    
    # Save overall summary
    with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create overall performance chart
    if model_performance:
        models = list(model_performance.keys())
        scores = list(model_performance.values())
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(models, scores)
        plt.title('Overall Model Performance Across All Language Pairs')
        plt.ylabel('Average Similarity Score (%)')
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{score:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_performance.png'), dpi=300)
        plt.close()
    
    return summary

def generate_report(input_dir="/content/africa-mt-benchmark/output", output_dir="/content/africa-mt-benchmark/reports"):
    """Main function to generate reports"""
    print("Generating performance reports...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results from all processed files
    results = collect_results(input_dir)
    
    if not results:
        print("No processed results found. Please run translations first.")
        return
    
    # Generate language-specific reports
    generate_language_specific_reports(results, output_dir)
    
    # Generate overall summary
    overall_summary = generate_overall_summary(results, output_dir)
    
    print(f"Reports generated successfully in {output_dir}/")
    
    if overall_summary:
        print(f"Overall best model: {overall_summary['best_overall_model']} ({overall_summary['best_overall_score']:.2f}%)")
    
    return results, overall_summary