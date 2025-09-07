import os
import pandas as pd
import importlib.util
from pathlib import Path
import re
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from reporting import generate_report

def load_recipes(recipes_dir="recipes"):
    recipes = {}
    for file in os.listdir(recipes_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(recipes_dir, file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            recipes[module_name] = module
    return recipes

def load_language_pairs(file_path="language_pairs.txt"):
    """Load language pairs from a text file"""
    language_pairs = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    if '-' in line:
                        source, target = line.split('-', 1)
                        language_pairs.append((source.strip(), target.strip()))
                    else:
                        print(f"Invalid language pair format: {line}")
        return language_pairs
    except FileNotFoundError:
        print(f"Language pairs file not found: {file_path}")
        return []

def process_csv(input_path, output_path, recipe_module, source_lang, target_lang):
    df = pd.read_csv(input_path)
    
    # Process with the specified language codes
    processed_df = recipe_module.process_dataframe(df, source_lang=source_lang, target_lang=target_lang)
    return processed_df

def get_output_filename(input_filename, recipe_name):
    """Generate output filename with recipe prefix"""
    name, ext = os.path.splitext(input_filename)
    return f"{name}_{recipe_name}{ext}"

def main():
    # Define input and output directories
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load language pairs from file
    language_pairs = load_language_pairs()
    if not language_pairs:
        print("No language pairs found. Please create a language_pairs.txt file.")
        return
    
    # Load recipes
    recipes = load_recipes()
    
    # Process each language pair
    for source_lang, target_lang in language_pairs:
        # Create language pair directory in output
        lang_pair_dir = os.path.join(output_dir, f"{source_lang}-{target_lang}")
        os.makedirs(lang_pair_dir, exist_ok=True)
        
        # Process each CSV file in the input directory
        for file in os.listdir(input_dir):
            if file.endswith(".csv"):
                input_path = os.path.join(input_dir, file)
                
                for recipe_name, recipe_module in recipes.items():
                    # Generate recipe-specific output filename
                    output_filename = get_output_filename(file, recipe_name)
                    output_path = os.path.join(lang_pair_dir, output_filename)
                    
                    # Check if this recipe has already processed this file for this language pair
                    if os.path.exists(output_path):
                        print(f"Skipping {recipe_name} for {file} ({source_lang}-{target_lang}) - already processed")
                        continue
                    
                    print(f"Processing {input_path} with recipe {recipe_name} for {source_lang}-{target_lang}")
                    print(f"Output will be saved to {output_path}")
                    
                    try:
                        result_df = process_csv(input_path, output_path, recipe_module, source_lang, target_lang)
                        result_df.to_csv(output_path, index=False)
                        print(f"Completed {recipe_name} on {file} for {source_lang}-{target_lang}")
                    except Exception as e:
                        print(f"Error applying {recipe_name} to {file} for {source_lang}-{target_lang}: {str(e)}")
    
    # Generate reports using the output directory
    generate_report(output_dir)

if __name__ == "__main__":
    main()