#judge.py


import os
import json
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# We need both sync and async clients now, ensure API key is loaded
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
    
    # The synchronous client for the original judge
    client = OpenAI(api_key=api_key)
    
    # The asynchronous client for our new multi-dimensional judge
    async_client = AsyncOpenAI(api_key=api_key)

except Exception as e:
    print(f"Error initializing OpenAI clients: {e}")
    exit()


DIMENSIONS_FILE_PATH = "data/evaluation_dimensions.json"

# --- The Original Judge (kept for legacy/simpler tasks) ---
def get_llm_judge_score(
    text_to_evaluate: str, 
    prompt_template: str, 
    model_name: str, 
    temperature: float
) -> int | None:
    """
    Calls an OpenAI LLM for a single numerical score.

    Args:
        text_to_evaluate: The text you want the LLM to score.
        prompt_template: A template string for the user prompt.
        model_name: The specific OpenAI model to use.
        temperature: The sampling temperature to use.

    Returns:
        An integer score if successful, None otherwise.
    """
    system_prompt = (
        "You are a helpful evaluation assistant. Your task is to rate the given text on a scale of 1 to 10. "
        "You must respond with ONLY a single integer and nothing else. Do not add any explanation or context."
    )
    
    user_prompt = prompt_template.format(text=text_to_evaluate)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5,
            temperature=temperature
        )
        
        response_text = response.choices[0].message.content.strip()
        return int(response_text)

    except Exception as e:
        print(f"An error occurred in get_llm_judge_score: {e}")
        return None


# --- NEW: The Multi-Dimensional Judge ---

def load_evaluation_dimensions() -> list:
    """Loads the AI-generated dimensions from the JSON file."""
    try:
        with open(DIMENSIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure we return a list, even if the file is empty
            dimensions = data.get("dimensions", [])
            if not isinstance(dimensions, list):
                print(f"Warning: 'dimensions' in {DIMENSIONS_FILE_PATH} is not a list.")
                return []
            return dimensions
    except FileNotFoundError:
        print(f"Error: Dimensions file not found at {DIMENSIONS_FILE_PATH}")
        print("Please run 'python scripts/suggest_dimensions.py' first.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DIMENSIONS_FILE_PATH}.")
        return []

async def _evaluate_single_dimension(
    dimension: dict, text_to_evaluate: str, model_name: str, temperature: float
) -> dict:
    """Async task to evaluate one dimension."""
    dimension_name = dimension.get('dimension_name', 'Unknown Dimension')
    description = dimension.get('description', 'No description provided.')

    # A generic, powerful prompt that uses the AI-generated dimension and description
    system_prompt = f"""
You are an evaluation expert. Your task is to evaluate a piece of text based on a specific dimension.
The dimension is '{dimension_name}': {description}.

Rate the text on a scale of 1 to 10 for this dimension.
Your final output MUST be a valid JSON object with your reasoning and score, like this:
{{"reasoning": "Your brief analysis here...", "score": integer}}
"""
    
    try:
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_evaluate}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        result_str = response.choices[0].message.content
        return json.loads(result_str)
    except Exception as e:
        print(f"Error evaluating dimension '{dimension_name}': {e}")
        return {"reasoning": f"API or Parse Error: {str(e)}", "score": None}

async def get_multidimensional_evaluation(
    text_to_evaluate: str, model_name: str, temperature: float
) -> dict:
    """
    Performs a multi-dimensional evaluation using AI-suggested criteria.
    """
    dimensions = load_evaluation_dimensions()
    if not dimensions:
        return {}

    # Create a parallel task for each dimension
    tasks = [
        _evaluate_single_dimension(dim, text_to_evaluate, model_name, temperature)
        for dim in dimensions
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Combine results into a final dictionary, using dimension_name as the key
    final_evaluation = {
        dimensions[i].get('dimension_name', f'unknown_{i}'): result 
        for i, result in enumerate(results)
    }
    
    return final_evaluation

# --- Test Block for the new judge ---
async def main():
    """
    A test function to demonstrate the new multi-dimensional judge.
    """
    print("--- Running Multi-Dimensional Judge Test ---")
    sample_text = "The sun, a star at the center of the Solar System, is a nearly perfect sphere of hot plasma. Earth and other matter orbit it."
    
    # We don't need a prompt; the judge gets the dimensions and their prompts from the file.
    evaluation = await get_multidimensional_evaluation(
        text_to_evaluate=sample_text,
        model_name="gpt-5.2",
        temperature=0.1
    )
    
    print("\n--- Evaluation Result ---")
    if evaluation:
        print(json.dumps(evaluation, indent=2))
    else:
        print("Evaluation failed. Did you run suggest_dimensions.py first?")
    print("--------------------------")

if __name__ == "__main__":
    asyncio.run(main())