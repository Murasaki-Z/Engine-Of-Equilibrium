# --- File: scripts/evaluate_config.py (Universal Wrapper) ---
import asyncio
import argparse
from measure_variance import measure_config_variance
from prepare_dataset import DATASET_REGISTRY

# --- Main execution block ---
async def main(dataset_name: str):
    """
    A wrapper to evaluate our champion configurations on a specific dataset.
    """
    print(f"===== Running Champion Evaluation on: {dataset_name} =====")
    
    # These were the champions for cnn_dailymail. They serve as our test candidates.
    # In a real run, we would discover new champions for each dataset.
    config_1 = {
        "model": "gpt-5.2",
        "temperature": 0.1,
        "prompt_style": "rubric" # prompt_style is not directly used by the new judge but we keep it for consistency
    }
    
    config_2 = {
        "model": "gpt-5.2",
        "temperature": 0.7,
        "prompt_style": "simple"
    }

    # Run the evaluation for the first configuration
    await measure_config_variance(config_1, "Low-Temp Champion", dataset_name)
    
    print("\n\n" + "="*50 + "\n\n") # Separator
    
    # Run the evaluation for the second configuration
    await measure_config_variance(config_2, "High-Temp Champion", dataset_name)
    
    print(f"\n===== Evaluation Complete for: {dataset_name} =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate champion configurations on a chosen dataset.")
    parser.add_argument(
        "dataset_name", 
        type=str, 
        choices=list(DATASET_REGISTRY.keys()),
        help="The name of the dataset to evaluate on."
    )
    args = parser.parse_args()
    
    asyncio.run(main(args.dataset_name))