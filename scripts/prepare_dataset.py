import os
import json
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
NUM_SAMPLES = 50  # The number of records for our golden dataset
OUTPUT_DIR = "data"
OUTPUT_FILENAME = "golden_dataset.json"

def create_golden_dataset():
    """
    Downloads a dataset from Hugging Face, processes a small sample,
    and saves it as our local 'golden_dataset.json'.
    """
    print(f"--- Starting dataset preparation for '{DATASET_NAME}' ---")

    try:
        # Load the dataset from Hugging Face. We'll use the 'test' split.
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split="test")
        print("Dataset loaded successfully.")

        # Ensure the output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        golden_samples = []
        print(f"Processing the first {NUM_SAMPLES} samples...")

        # Take the first N samples for our static dataset
        for i in range(NUM_SAMPLES):
            sample = dataset[i]
            processed_sample = {
                "id": sample["id"],
                # The text to be summarized is the 'article'
                "source_text": sample["article"],
                # The text we will have the judge evaluate is the 'highlights'
                "text_to_judge": sample["highlights"]
            }
            
            golden_samples.append(processed_sample)
        
        print(f"Processed {len(golden_samples)} samples.")

        # Define the full path for the output file
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

        # Write the processed samples to a JSON file
        print(f"Saving golden dataset to '{output_path}'...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(golden_samples, f, ensure_ascii=False, indent=4)
        
        print("--- Golden dataset created successfully! ---")

    except Exception as e:
        print(f"An error occurred during dataset preparation: {e}")

# --- Run the script ---
if __name__ == "__main__":
    create_golden_dataset()