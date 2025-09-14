import os
import pandas as pd
import importlib.util
from pathlib import Path
import re
import sys
import json

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

def load_language_pairs(input_dir="input"):
    """Load language pairs from language_pairs.txt in the input directory"""
    language_pairs = []
    file_path = os.path.join(input_dir, "language_pairs.txt")
    
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

def load_processing_state(state_file="processing_state.json"):
    """Load the processing state from a JSON file"""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_processing_state(state, state_file="processing_state.json"):
    """Save the processing state to a JSON file"""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def process_csv(input_path, recipe_module, source_lang, target_lang, mode="full"):
    df = pd.read_csv(input_path)
    
    # Process with the specified language codes
    if mode == "forward_only" and hasattr(recipe_module, 'forward_translation_only'):
        processed_df = recipe_module.forward_translation_only(df, source_lang=source_lang, target_lang=target_lang)
    elif mode == "backtranslation_only" and hasattr(recipe_module, 'backtranslation_only'):
        processed_df = recipe_module.backtranslation_only(df, source_lang=source_lang, target_lang=target_lang)
    else:
        processed_df = recipe_module.process_dataframe(df, source_lang=source_lang, target_lang=target_lang)
    
    return processed_df

def get_output_filename(input_filename, recipe_name):
    """Generate output filename with recipe prefix"""
    name, ext = os.path.splitext(input_filename)
    return f"{name}_{recipe_name}{ext}"

def run_forward_translation(input_dir, output_dir, language_pairs, recipes, state):
    """Run only the forward translation part"""
    print("Running forward translation only...")
    
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
                    
                    # Check if this recipe has already completed forward translation for this file
                    state_key = f"{source_lang}-{target_lang}/{file}/{recipe_name}"
                    if state.get(state_key, {}).get('forward_completed', False):
                        print(f"Skipping forward translation for {recipe_name} on {file} ({source_lang}-{target_lang}) - already completed")
                        continue
                    
                    print(f"Processing {input_path} with recipe {recipe_name} for {source_lang}-{target_lang}")
                    
                    try:
                        # Check if recipe supports forward translation only mode
                        if hasattr(recipe_module, 'forward_translation_only'):
                            result_df = process_csv(input_path, recipe_module, 
                                                  source_lang, target_lang, "forward_only")
                            result_df.to_csv(output_path, index=False)
                            
                            # Update state
                            if state_key not in state:
                                state[state_key] = {}
                            state[state_key]['forward_completed'] = True
                            save_processing_state(state)
                            
                            print(f"Completed forward translation with {recipe_name} on {file} for {source_lang}-{target_lang}")
                        else:
                            print(f"Recipe {recipe_name} doesn't support forward-only mode")
                    except Exception as e:
                        print(f"Error applying {recipe_name} to {file} for {source_lang}-{target_lang}: {str(e)}")

def run_backtranslation(input_dir, output_dir, language_pairs, recipes, state):
    """Run only the backtranslation and similarity part"""
    print("Running backtranslation and similarity only...")
    
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
                    
                    # Check if forward translation has been completed
                    state_key = f"{source_lang}-{target_lang}/{file}/{recipe_name}"
                    if not state.get(state_key, {}).get('forward_completed', False):
                        print(f"Skipping backtranslation for {recipe_name} on {file} ({source_lang}-{target_lang}) - forward translation not completed")
                        continue
                    
                    # Check if backtranslation has already been completed
                    if state.get(state_key, {}).get('backtranslation_completed', False):
                        print(f"Skipping backtranslation for {recipe_name} on {file} ({source_lang}-{target_lang}) - already completed")
                        continue
                    
                    print(f"Processing backtranslation for {recipe_name} on {file} ({source_lang}-{target_lang})")
                    
                    try:
                        # Check if recipe supports backtranslation only mode
                        if hasattr(recipe_module, 'backtranslation_only'):
                            # Read the file that should contain forward translations
                            if os.path.exists(output_path):
                                # Load backtranslation models only when needed
                                if hasattr(recipe_module, 'load_backtranslation_models'):
                                    recipe_module.load_backtranslation_models()
                                
                                df = pd.read_csv(output_path)
                                result_df = process_csv(output_path, recipe_module, 
                                                      source_lang, target_lang, "backtranslation_only")
                                result_df.to_csv(output_path, index=False)
                                
                                # Update state
                                state[state_key]['backtranslation_completed'] = True
                                save_processing_state(state)
                                
                                print(f"Completed backtranslation with {recipe_name} on {file} for {source_lang}-{target_lang}")
                            else:
                                print(f"File not found: {output_path}")
                        else:
                            print(f"Recipe {recipe_name} doesn't support backtranslation-only mode")
                    except Exception as e:
                        print(f"Error applying backtranslation with {recipe_name} to {file} for {source_lang}-{target_lang}: {str(e)}")

def run_full_process(input_dir, output_dir, language_pairs, recipes, state):
    """Run the full process (forward translation + backtranslation + similarity)"""
    print("Running full process...")
    
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
                    state_key = f"{source_lang}-{target_lang}/{file}/{recipe_name}"
                    if state.get(state_key, {}).get('backtranslation_completed', False):
                        print(f"Skipping {recipe_name} for {file} ({source_lang}-{target_lang}) - already processed")
                        continue
                    
                    print(f"Processing {input_path} with recipe {recipe_name} for {source_lang}-{target_lang}")
                    print(f"Output will be saved to {output_path}")
                    
                    try:
                        # For full process, load backtranslation models only when needed
                        if hasattr(recipe_module, 'load_backtranslation_models'):
                            recipe_module.load_backtranslation_models()
                            
                        result_df = process_csv(input_path, recipe_module, source_lang, target_lang)
                        result_df.to_csv(output_path, index=False)
                        
                        # Update state
                        state[state_key] = {
                            'forward_completed': True,
                            'backtranslation_completed': True
                        }
                        save_processing_state(state)
                        
                        print(f"Completed {recipe_name} on {file} for {source_lang}-{target_lang}")
                    except Exception as e:
                        print(f"Error applying {recipe_name} to {file} for {source_lang}-{target_lang}: {str(e)}")

def display_menu():
    """Display the menu options"""
    print("\n" + "="*50)
    print("Translation Pipeline Menu")
    print("="*50)
    print("1. Run only forward translation")
    print("2. Run only backtranslation and similarity")
    print("3. Run full process (forward + backtranslation + similarity)")
    print("4. Generate reports only")
    print("5. Reset processing state")
    print("6. Exit")
    print("="*50)
    
    while True:
        try:
            choice = input("Please select an option (1-6): ")
            if choice in ["1", "2", "3", "4", "5", "6"]:
                return choice
            else:
                print("Invalid option. Please enter a number between 1 and 6.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def reset_processing_state(state_file="processing_state.json"):
    """Reset the processing state"""
    if os.path.exists(state_file):
        os.remove(state_file)
    print("Processing state has been reset.")

def main():
    # Define input and output directories
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load language pairs from file in input directory
    language_pairs = load_language_pairs(input_dir)
    if not language_pairs:
        print("No language pairs found. Please create a language_pairs.txt file in the input directory.")
        return
    
    # Load recipes
    recipes = load_recipes()
    
    # Load processing state
    state = load_processing_state()
    
    while True:
        choice = display_menu()
        
        if choice == "1":
            run_forward_translation(input_dir, output_dir, language_pairs, recipes, state)
        elif choice == "2":
            run_backtranslation(input_dir, output_dir, language_pairs, recipes, state)
        elif choice == "3":
            run_full_process(input_dir, output_dir, language_pairs, recipes, state)
        elif choice == "4":
            generate_report(output_dir)
        elif choice == "5":
            reset_processing_state()
            state = {}  # Reset in-memory state
        elif choice == "6":
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
