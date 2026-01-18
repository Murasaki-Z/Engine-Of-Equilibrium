# --- File: scripts/prepare_dataset.py ---
import os
import json
import argparse
from datasets import load_dataset
from get_dimensions import generate_and_save_dimensions
import random

# --- The Dataset Registry ---
# A mapping of dataset names to their specific configurations.
# This makes our script adaptable to different data structures.
DATASET_REGISTRY = {
    "cnn_dailymail": {
        "hf_name": "cnn_dailymail",
        "hf_version": "3.0.0",
        "split": "test",
        "column_to_judge": "highlights",
        "column_source": "article" # Optional source text
    },
    "xsum": {
        "hf_name": "xsum",
        "hf_version": None, # Let Hugging Face decide the default version
        "split": "test",
        "column_to_judge": "summary",
        "column_source": "document"
    },
    "squad": {
        "hf_name": "squad",
        "hf_version": None,
        "split": "validation",
        "column_to_judge": "answers", # Note: this column is a dictionary itself
        "column_source": "question"
    },
    "imdb": {
        "hf_name": "imdb",
        "hf_version": None,
        "split": "test",
        "column_to_judge": "text", # Here we'd be judging the sentiment/quality of the review text
        "column_source": None # No source text needed
    },
    "medical_questions_pairs": {
        "hf_name": "medical_questions_pairs",
        "hf_version": None,
        "split": "train", # This dataset only has a 'train' split
        "column_to_judge": "question_1",
        "column_source": "question_2"
    }
}

# --- Configuration ---
NUM_SAMPLES = 50
OUTPUT_DIR = "data"

def prepare_dataset_for(dataset_name: str):
    """
    A universal function to download any registered dataset, save a local sample,
    and generate the evaluation dimensions for it.
    
    Args:
        dataset_name (str): The short name of the dataset from our registry.
    """
    if dataset_name not in DATASET_REGISTRY:
        print(f"Error: Dataset '{dataset_name}' not found in our registry.")
        print(f"Available datasets: {list(DATASET_REGISTRY.keys())}")
        return

    config = DATASET_REGISTRY[dataset_name]
    print(f"--- 🚀 Starting Full Preparation for: '{dataset_name}' ---")

    try:
        # --- PART 1: PREPARE THE DATASET ---
        print(f"\n[Step 1/2] Loading '{config['hf_name']}' from Hugging Face...")
        dataset = load_dataset(config['hf_name'], config['hf_version'], split=config['split'])
        print("Dataset loaded successfully.")

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        golden_samples = []
        print(f"Processing the first {NUM_SAMPLES} samples...")
        
        for i in range(NUM_SAMPLES):
            sample = dataset[i]
            
            # --- Smart column extraction ---
            text_to_judge = sample[config['column_to_judge']]
            # Handle cases like SQuAD where the column is a dict
            if isinstance(text_to_judge, dict):
                text_to_judge = text_to_judge.get('text', [''])[0]

            source_text = sample[config['column_source']] if config['column_source'] else None
            
            processed_sample = {
                "id": f"{dataset_name}_{i}",
                "source_text": str(source_text),
                "text_to_judge": str(text_to_judge)
            }
            golden_samples.append(processed_sample)
        
        # Save the dataset with a specific name
        output_filename = f"{dataset_name}_golden_dataset.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        print(f"Saving golden dataset to '{output_path}'...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(golden_samples, f, ensure_ascii=False, indent=4)
        print("✅ Golden dataset created successfully.")

        # --- PART 2: GENERATE THE DIMENSIONS ---
        print("\n[Step 2/2] Generating evaluation dimensions for this dataset...")
        # We now save dimensions to a dataset-specific file as well
        generate_and_save_dimensions(golden_samples, dataset_name)
        
        print(f"\n--- ✅ Full Preparation for '{dataset_name}' Complete! ---")

    except Exception as e:
        print(f"An error occurred during dataset preparation: {e}")

# --- Command-Line Interface ---
if __name__ == "__main__":
    # We use argparse to let the user choose the dataset from the command line
    parser = argparse.ArgumentParser(description="Prepare a dataset for the Engine of Equilibrium.")
    parser.add_argument(
        "dataset_name", 
        type=str, 
        nargs="?",
        default=None, 
        choices=list(DATASET_REGISTRY.keys()),
        help="The name of the dataset to prepare."
    )
    args = parser.parse_args()

    if args.dataset_name:
        # If the user provided a name, use it
        selected_dataset = args.dataset_name
        print(f"User selected dataset: '{selected_dataset}'")
    else:
        # If no name was provided, pick one at random
        selected_dataset = random.choice(list(DATASET_REGISTRY.keys()))
        print(f"No dataset specified. Randomly selected: '{selected_dataset}'")
    
    prepare_dataset_for(selected_dataset)
