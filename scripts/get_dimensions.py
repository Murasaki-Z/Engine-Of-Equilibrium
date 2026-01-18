# dimensions.py
import json
import os
import sys

# Allow this script to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.judge import client as openai_client # Import the synchronous client

# --- Configuration ---
OUTPUT_PATH = "data/evaluation_dimensions.json"
NUM_SAMPLES_FOR_ANALYSIS = 5

# --- The core logic is now a reusable function ---
def generate_and_save_dimensions(dataset: list, dataset_name: str):
    """
    Uses an LLM to analyze dataset samples and saves relevant evaluation dimensions.
    
    Args:
        dataset (list): A list of sample dictionaries from the golden dataset.
    """
    print("\n--- 🧠 Starting AI Dimension Suggester ---")
    
    if not dataset:
        print("Dataset is empty. Cannot suggest dimensions.")
        return
        
    samples_to_analyze = dataset[:NUM_SAMPLES_FOR_ANALYSIS]
    prompt_samples = [{"text_to_judge": sample['text_to_judge']} for sample in samples_to_analyze]

    system_prompt = """
You are an expert in data analysis and evaluation science. Your task is to propose a set of relevant evaluation dimensions for the given dataset samples.
Instructions:
1. Analyze the following text samples to understand their domain, purpose, and style.
2. Based on your analysis, propose 3 to 5 distinct, high-level dimensions that would be most effective for evaluating the quality of similar texts.
3. For each dimension, provide a one-sentence description.
4. Your final output MUST be a valid JSON object, adhering to the following schema:
   {"dimensions": [{"dimension_name": "string", "description": "string"}]}
"""
    user_prompt = f"Here are the data samples to analyze:\n```json\n{json.dumps(prompt_samples, indent=2)}\n```"

    print("Calling LLM to suggest dimensions...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_json_str = response.choices[0].message.content
        suggested_data = json.loads(response_json_str)

        if "dimensions" not in suggested_data or not isinstance(suggested_data["dimensions"], list):
            raise ValueError("LLM output is missing the 'dimensions' list.")

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(suggested_data, f, indent=4)
            
        print(f"✅ Dimensions saved successfully to '{OUTPUT_PATH}'")
        print("Generated Dimensions:")
        print(json.dumps(suggested_data, indent=2))

    except Exception as e:
        print(f"--- 🚫 An error occurred ---")
        print(f"Failed to generate dimensions: {e}")

# --- Main block for standalone testing ---
if __name__ == "__main__":
    # You can still run this file directly to test the dimension suggestion
    DATASET_PATH = "data/golden_dataset.json"
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
        generate_and_save_dimensions(full_dataset)
    except FileNotFoundError:
        print(f"Error: {DATASET_PATH} not found. Please run prepare_dataset.py first.")