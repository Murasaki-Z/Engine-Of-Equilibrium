# --- File: scripts/measure_variance.py ---
import json
import numpy as np
import sys
import os
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.judge import get_llm_judge_score

# --- Configuration ---
DATASET_PATH = "data/golden_dataset.json"
NUM_SAMPLES_TO_TEST = 10
NUM_RUNS_PER_SAMPLE = 10

PROMPT_TEMPLATES = {
    "simple": "On a scale of 1 to 10, how would you rate the quality of this summary? Text: '{text}'",
    "rubric": """
Rate the quality of the following summary on a scale of 1 to 10 based on the rubric below.
**Rubric:**
- **Clarity (1-10):** Is the summary easy to understand?
- **Accuracy (1-10):** Does the summary correctly represent the key information from the source text? (Source not provided, assess based on internal consistency and factual plausibility).
**Task:**
Provide a single, overall score from 1 to 10.
**Summary to Evaluate:**
'{text}'
"""
}

def load_dataset(path: str) -> list:
    """Loads the golden dataset from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{path}'.")
        return None

# --- Main logic is now a reusable async function ---
async def measure_config_variance(config: dict, config_name: str) -> float | None:
    """
    Measures the average variance of a given LLM judge configuration.
    """
    print(f"--- 📏 Starting Evaluation for: '{config_name}' ---")
    print(f"Configuration: {json.dumps(config, indent=4)}")

    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        return None

    # Get parameters from the config dictionary
    model_name = config['model']
    temperature = config['temperature']
    prompt_template = PROMPT_TEMPLATES[config['prompt_style']]
    
    all_variances = []
    samples_to_process = dataset[:NUM_SAMPLES_TO_TEST]

    for i, sample in enumerate(samples_to_process):
        sample_id = sample['id']
        text_to_judge = sample['text_to_judge']
        
        print(f"\nProcessing sample {i+1}/{len(samples_to_process)} (ID: {sample_id})...")

        # Define the async task for getting a single score
        async def get_score_task(run_num):
            # asyncio.to_thread runs our synchronous judge function in a separate thread
            score = await asyncio.to_thread(
                get_llm_judge_score,
                text_to_judge,
                prompt_template,
                model_name,
                temperature
            )
            if score is not None:
                print(f"  Run {run_num}/{NUM_RUNS_PER_SAMPLE}: Score = {score}")
            else:
                print(f"  Run {run_num}/{NUM_RUNS_PER_SAMPLE}: Failed.")
            return score

        # Call the judge N times concurrently
        tasks = [get_score_task(j+1) for j in range(NUM_RUNS_PER_SAMPLE)]
        results = await asyncio.gather(*tasks)
        scores = [s for s in results if s is not None]

        if len(scores) > 1:
            sample_variance = np.var(scores, ddof=1)
            all_variances.append(sample_variance)
            print(f"-> Scores for this sample: {scores}")
            print(f"-> Variance for this sample: {sample_variance:.4f}")

    if all_variances:
        average_variance = np.mean(all_variances)
        print("\n--- 📊 Evaluation Complete ---")
        print(f"Configuration Tested: '{config_name}'")
        print(f"Final Average Variance: {average_variance:.4f}")
        print("----------------------------")
        return average_variance
    else:
        print("\n--- Measurement Failed ---")
        return None

# --- The original baseline script functionality ---
async def run_baseline_test():
    """Runs the test for the original, simple baseline configuration."""
    baseline_config = {
        "model": "gpt-5.2", # Or whatever model you used for the original baseline
        "temperature": 0.7,
        "prompt_style": "simple"
    }
    await measure_config_variance(baseline_config, "Original Baseline")

if __name__ == "__main__":
    # This script can now be run to re-calculate the original baseline
    asyncio.run(run_baseline_test())