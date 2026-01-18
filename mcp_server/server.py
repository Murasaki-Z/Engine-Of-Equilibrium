import numpy as np
import sys
import os
from typing import Dict, List, Any

# Allow the server to import our judge function from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.judge import get_llm_judge_score
from fastmcp import FastMCP

# --- Server Setup ---
server = FastMCP(name="Taguchi Optimization Server")

# --- Experiment Definitions ---
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

# --- MCP Tools ---

@server.tool()
async def generate_taguchi_design() -> Dict[str, Any]:
    """
    Generates a pre-defined L4 Orthogonal Array for a 3-factor, 2-level experiment.
    """
    design = {
        "factors": ["model", "temperature", "prompt_style"],
        "design_array": [
            {"run": 1, "config": {"model": "gpt-4o", "temperature": 0.1, "prompt_style": "simple"}},
            {"run": 2, "config": {"model": "gpt-4o", "temperature": 0.7, "prompt_style": "rubric"}},
            {"run": 3, "config": {"model": "gpt-5.2", "temperature": 0.1, "prompt_style": "rubric"}},
            {"run": 4, "config": {"model": "gpt-5.2", "temperature": 0.7, "prompt_style": "simple"}},
        ]
    }
    return design

@server.tool()
async def run_judge_trials(design_array: List[Dict[str, Any]], text_to_judge: str, n_runs_per_sample: int = 10) -> Dict[str, Any]:
    """
    Executes the Taguchi experiment.
    """
    trial_results = {}
    for experiment in design_array:
        run_id = experiment['run']
        config = experiment['config']
        
        model = config['model']
        temperature = config['temperature']
        prompt_template_key = config['prompt_style']
        prompt_template = PROMPT_TEMPLATES[prompt_template_key]
        
        scores = []
        print(f"\n--- Running Experiment {run_id}: {config} ---")
        for i in range(n_runs_per_sample):
            score = get_llm_judge_score(
                text_to_evaluate=text_to_judge,
                prompt_template=prompt_template,
                model_name=model,
                temperature=temperature
            )
            if score is not None:
                scores.append(score)
                print(f"  Run {i+1}/{n_runs_per_sample}: Score = {score}")
            else:
                print(f"  Run {i+1}/{n_runs_per_sample}: Failed")
        trial_results[f"run_{run_id}"] = {"config": config, "scores": scores}
    return trial_results

@server.tool()
async def calculate_robustness(trial_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the trial results to find the most robust configuration.
    """
    analysis = {}
    for run_id, results in trial_results.items():
        scores = results['scores']
        if len(scores) < 2:
            continue
            
        variance = np.var(scores, ddof=1)
        mean_of_squares = np.mean(np.square(scores))
        sn_ratio = -10 * np.log10(mean_of_squares) if mean_of_squares > 0 else 0
        
        analysis[run_id] = {
            "config": results['config'],
            "variance": variance,
            "sn_ratio": sn_ratio
        }
    
    best_run = max(analysis.items(), key=lambda item: item[1]['sn_ratio']) if analysis else (None, {})
    
    return {
        "full_analysis": analysis,
        "optimal_configuration": best_run[1].get('config', 'N/A'),
        "best_sn_ratio": best_run[1].get('sn_ratio', 'N/A')
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Engine of Equilibrium: Taguchi Toolkit MCP Server ---")
    print("Starting MCP server with SSE transport on 0.0.0.0:8000")
    print("Available tools: generate_taguchi_design, run_judge_trials, calculate_robustness")
    
    server.run(transport="sse", host="0.0.0.0", port=8000)