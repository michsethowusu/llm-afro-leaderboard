import os
import pandas as pd
import importlib.util
from pathlib import Path
import re
import sys
import shutil

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '/content/africa-mt-benchmark/utils'))
from reporting import generate_report

def load_recipes(recipes_dir="/content/africa-mt-benchmark/recipes"):
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

def extract_lang_codes_from_path(path):
    """Extract language codes from folder path like 'eng-twi'"""
    # Get the parent folder name
    folder_name = os.path.basename(os.path.dirname(path))
    
    # Match patterns like 'eng-twi' or 'fra-yor'
    match = re.match(r'^([a-z]{2,3})-([a-z]{2,3})$', folder_name)
    if match:
        source_code, target_code = match.groups()
        return source_code, target_code
    return None, None

def process_csv(input_path, output_path, recipe_module):
    df = pd.read_csv(input_path)
    
    # Extract language codes from folder path
    source_code, target_code = extract_lang_codes_from_path(input_path)
    
    if source_code and target_code:
        # Process with the detected language codes
        processed_df = recipe_module.process_dataframe(df, source_lang=source_code, target_lang=target_code)
        return processed_df
    else:
        print(f"Could not extract language codes from path: {input_path}")
        return df

def main():
    # Define input and output directories
    input_dir = "/content/africa-mt-benchmark/input"
    output_dir = "/content/africa-mt-benchmark/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    recipes = load_recipes()
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                input_path = os.path.join(root, file)
                
                # Create corresponding output path
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, file)
                
                print(f"Processing {input_path}")
                print(f"Output will be saved to {output_path}")
                
                for recipe_name, recipe_module in recipes.items():
                    print(f"Applying recipe: {recipe_name}")
                    try:
                        result_df = process_csv(input_path, output_path, recipe_module)
                        result_df.to_csv(output_path, index=False)
                        print(f"Completed {recipe_name} on {file}")
                    except Exception as e:
                        print(f"Error applying {recipe_name} to {file}: {str(e)}")
    
    # Generate reports using the output directory
    generate_report(output_dir)

if __name__ == "__main__":
    main()
