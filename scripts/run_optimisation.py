# --- File: scripts/run_optimization.py ---
import asyncio
import json
import sys
import os

# Allow this script to see the 'data' directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Configuration ---
SERVER_URL = "http://localhost:8000/sse/"
DATASET_PATH = "data/golden_dataset.json"

def load_first_sample(path: str) -> dict:
    """Loads just the first sample from our golden dataset for the experiment."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            return dataset[0]
    except (FileNotFoundError, IndexError):
        print(f"Error: Dataset not found or is empty at '{path}'.")
        return None

async def run_full_experiment():
    """
    Orchestrates the entire Taguchi optimization experiment by calling the MCP server.
    """
    print("--- 🚀 Starting Optimization Experiment ---")

    # 1. Load data
    print("\n[Step 1/4] Loading a sample from the Golden Dataset...")
    sample = load_first_sample(DATASET_PATH)
    if not sample:
        return
    text_to_judge = sample['text_to_judge']
    print("  -> Sample loaded successfully.")

    # Import here to avoid issues if fastmcp not installed
    from fastmcp import Client
    
    client = Client(SERVER_URL)
    
    async with client:
        # 2. Get the experimental plan
        print("\n[Step 2/4] Calling 'generate_taguchi_design' tool...")
        design_response = await client.call_tool("generate_taguchi_design", {})
        design_data = json.loads(design_response.content[0].text)
        design_array = design_data["design_array"]
        print("  -> Received L4 Orthogonal Array design.")

        # 3. Execute the trials
        print("\n[Step 3/4] Calling 'run_judge_trials' tool...")
        print("  -> This will take a few minutes as it makes 40 API calls...")
        trial_response = await client.call_tool("run_judge_trials", {
            "design_array": design_array,
            "text_to_judge": text_to_judge
        })
        trial_results = json.loads(trial_response.content[0].text)
        print("  -> All trials completed successfully.")

        # 4. Analyze the results
        print("\n[Step 4/4] Calling 'calculate_robustness' tool...")
        analysis_response = await client.call_tool("calculate_robustness", {
            "trial_results": trial_results
        })
        final_analysis = json.loads(analysis_response.content[0].text)
        print("  -> Analysis complete.")

    # 5. Print the final report (outside the async block)
    print("\n--- ✅ Experiment Complete: Final Report ---")
    print("\nFull Analysis:")
    for run_id, analysis in final_analysis.get("full_analysis", {}).items():
        config = analysis.get('config')
        variance = analysis.get('variance')
        sn_ratio = analysis.get('sn_ratio')
        print(f"  - {run_id}: Var={variance:.4f}, S/N Ratio={sn_ratio:.4f} | Config: {config}")
    
    print("\nOptimal Configuration Found:")
    optimal_config = final_analysis.get('optimal_configuration')
    print(json.dumps(optimal_config, indent=4))
    print("\n--- Mission Accomplished ---")


if __name__ == "__main__":
    # asyncio.run() is needed to execute the async main function
    asyncio.run(run_full_experiment())