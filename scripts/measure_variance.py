# --- File: scripts/measure_variance.py (Universal Version) ---
import json
import numpy as np
import sys
import os
import asyncio
import argparse
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# We now import the multi-dimensional judge
from src.judge import get_multidimensional_evaluation

# --- Configuration ---
NUM_SAMPLES_TO_TEST = 10
NUM_RUNS_PER_SAMPLE = 10 # Lowered for speed during refactoring, can be increased later

def load_dataset(dataset_name: str) -> list:
    """Loads the specified golden dataset from a JSON file."""
    path = f"data/{dataset_name}_golden_dataset.json"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{path}'.")
        print(f"Please run 'python scripts/prepare_dataset.py {dataset_name}' first.")
        return None

# --- The core logic, now fully multi-dimensional and dataset-aware ---
async def measure_config_variance(config: dict, config_name: str, dataset_name: str) -> float | None:
    """
    Measures the average dimensional variance of a given LLM judge configuration
    for a specific dataset.
    """
    print(f"--- 📏 Starting Evaluation for: '{config_name}' on dataset '{dataset_name}' ---")
    print(f"Configuration: {json.dumps(config, indent=4)}")

    dataset = load_dataset(dataset_name)
    if not dataset:
        return None

    # Get parameters from the config dictionary
    model_name = config['model']
    temperature = config['temperature']
    
    # Store the variances for each dimension separately
    # e.g., {"Clarity_variance": [0.1, 0.2], "Accuracy_variance": [0.0, 0.1]}
    dimensional_variances = defaultdict(list)
    
    samples_to_process = dataset[:NUM_SAMPLES_TO_TEST]

    for i, sample in enumerate(samples_to_process):
        sample_id = sample['id']
        text_to_judge = sample['text_to_judge']
        
        print(f"\nProcessing sample {i+1}/{len(samples_to_process)} (ID: {sample_id})...")

        # --- ASYNC MULTI-DIMENSIONAL EVALUATION ---
        async def get_evaluation_task(run_num):
            eval_result = await get_multidimensional_evaluation(
                text_to_judge,
                model_name,
                temperature
            )
            print(f"  Run {run_num}/{NUM_RUNS_PER_SAMPLE} complete.")
            return eval_result

        # Call the multi-dimensional judge N times concurrently
        tasks = [get_evaluation_task(j+1) for j in range(NUM_RUNS_PER_SAMPLE)]
        results = await asyncio.gather(*tasks)
        
        # --- NEW: Process multi-dimensional results ---
        # Reorganize results for easier variance calculation
        # From: [eval1, eval2, ...] where eval = {"Clarity": {"score": 8}, "Accuracy": ...}
        # To:   {"Clarity": [8, 9, 8], "Accuracy": [7, 7, 7]}
        scores_by_dimension = defaultdict(list)
        for evaluation in results:
            for dim, result in evaluation.items():
                score = result.get('score')
                if score is not None and isinstance(score, (int, float)):
                    scores_by_dimension[dim].append(score)

        # Calculate variance for each dimension for this sample
        print("-> Variances for this sample:")
        for dim, scores in scores_by_dimension.items():
            if len(scores) > 1:
                sample_variance = np.var(scores, ddof=1)
                dimensional_variances[dim].append(sample_variance)
                print(f"    - {dim}: {sample_variance:.4f} (Scores: {scores})")

    # --- NEW: Calculate the final average across all dimensions and samples ---
    all_variance_values = [var for variances in dimensional_variances.values() for var in variances]

    if all_variance_values:
        average_dimensional_variance = np.mean(all_variance_values)
        print("\n--- 📊 Evaluation Complete ---")
        print(f"Configuration Tested: '{config_name}' on '{dataset_name}'")
        print(f"Final Average Dimensional Variance: {average_dimensional_variance:.4f}")
        print("---------------------------------")
        return average_dimensional_variance
    else:
        print("\n--- Measurement Failed: No valid scores were collected. ---")
        return None

# --- Main block for establishing a baseline on a chosen dataset ---
if __name__ == "__main__":
    from prepare_dataset import DATASET_REGISTRY # Import to know available choices
    
    parser = argparse.ArgumentParser(description="Measure the baseline variance for a given dataset.")
    parser.add_argument(
        "dataset_name", 
        type=str, 
        choices=list(DATASET_REGISTRY.keys()),
        help="The name of the dataset to measure."
    )
    args = parser.parse_args()

    # A standard, simple configuration to serve as our baseline for any dataset
    baseline_config = {
        "model": "gpt-5.2",
        "temperature": 0.7,
        "prompt_style": "rubric" # Using rubric as a sensible default
    }
    
    asyncio.run(measure_config_variance(baseline_config, f"Baseline for {args.dataset_name}", args.dataset_name))