import os
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"Error: Failed to instantiate OpenAI client. Is your API key set in .env?")
    print(e)
    exit()

# --- The Core Function (Upgraded) ---
def get_llm_judge_score(
    text_to_evaluate: str,
    prompt_template: str,
    model_name: str = "gpt-5.2",
    temperature: float = 1
) -> int | None:
    """
    Calls an OpenAI LLM using the Responses API to judge a piece of text.

    Args:
        text_to_evaluate: The text you want the LLM to score.
        prompt_template: A template string for the user prompt.
        model_name: The specific OpenAI model to use.
        temperature: The sampling temperature to use (controls randomness).

    Returns:
        An integer score if successful, None otherwise.
    """
    system_prompt = (
        "You are a helpful evaluation assistant. Your task is to rate the given text on a scale of 1 to 10. "
        "You must respond with ONLY a single integer and nothing else. Do not add any explanation or context."
    )
    user_prompt = prompt_template.format(text=text_to_evaluate)

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
            ],
            instructions=system_prompt,
            max_output_tokens=16,
            temperature=temperature
        )
        
        response_text = response.output[0].content[0].text.strip()
        
        return int(response_text)

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return None

# --- Test Block ---
if __name__ == "__main__":
    print("--- Running MVP Judge Test (Responses API & GPT-5.2) ---")

    sample_text = "The sun is a star. It is very big and hot. Earth orbits the sun."
    sample_prompt = "How would you rate the factual accuracy and clarity of this summary from 1-10? Text: '{text}'"
    
    # Call our refactored judge function with a specific model and temperature
    score = get_llm_judge_score(
        text_to_evaluate=sample_text,
        prompt_template=sample_prompt,
        model_name="gpt-5.2",
        temperature=0.1
    )

    if score is not None:
        print(f"Text: '{sample_text}'")
        print(f"Judged Score: {score}")
        print("--- Test Successful ---")
    else:
        print("--- Test Failed ---")