import os
import re

# Paths
base_script = "models_nvidia.py"
models_file = "models_nvidia.txt"
output_dir = "per_model_scripts"

# Read models
with open(models_file, "r") as f:
    models = [line.strip() for line in f if line.strip() and not line.startswith("#")]

os.makedirs(output_dir, exist_ok=True)

# Read base code
with open(base_script, "r") as f:
    base_code = f.read()

# Regex to strip out the whole get_model_list function
pattern = re.compile(r"def get_model_list\([\s\S]*?return models", re.MULTILINE)

for model in models:
    safe_model_name = model.replace("/", "_").replace("-", "_")
    output_file = os.path.join(output_dir, f"run_{safe_model_name}.py")

    # Remove the old get_model_list completely
    code_no_list = re.sub(pattern, "", base_code)

    # Add new minimal get_model_list
    custom_code = code_no_list + f"""

def get_model_list():
    return ['{model}']  # Single model only

if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({{'text': ["Hello world", "Testing translation"]}})
    results = process_dataframe(df, source_lang="en", target_lang="fr")
    print(results.head())
"""

    with open(output_file, "w") as f:
        f.write(custom_code)

print(f"✅ Generated {len(models)} scripts in {output_dir}/")

