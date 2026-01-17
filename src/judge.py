import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"Error: Failed to instantiate OpenAI client. Is your API key set in .env?")
    print(e)
    exit()

# --- The Core Function ---
def get_llm_judge_score(text_to_evaluate: str, prompt_template: str) -> int | None:
    """
    Calls an OpenAI LLM to judge a piece of text based on a prompt and returns a numerical score.

    Args:
        text_to_evaluate: The text you want the LLM to score.
        prompt_template: A template string that will be used for the prompt.

    Returns:
        An integer score if successful, None otherwise.
    """
    # The system prompt is now part of the 'messages' list for OpenAI
    system_prompt = (
        "You are a helpful evaluation assistant. Your task is to rate the given text on a scale of 1 to 10. "
        "You must respond with ONLY a single integer and nothing else. Do not add any explanation or context."
    )
    
    # Format the user's prompt using the text we want to evaluate.
    user_prompt = prompt_template.format(text=text_to_evaluate)

    try:
        # Note the change to client.chat.completions.create
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5,
            temperature=0.7
        )
        
        # The response structure is different for OpenAI
        response_text = response.choices[0].message.content.strip()
        
        # Safely convert the response to an integer.
        return int(response_text)

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return None

# --- Test Block ---
if __name__ == "__main__":
    print("--- Running MVP Judge Test (OpenAI Backend) ---")

    sample_text = "The sun is a star. It is very big and hot. Earth goes around the sun."

    sample_prompt = "How good is the following summary? Here is the text: '{text}'"
    
    score = get_llm_judge_score(sample_text, sample_prompt)

    if score is not None:
        print(f"Text: '{sample_text}'")
        print(f"Prompt: '{sample_prompt}'")
        print(f"Judged Score: {score}")
        print("--- Test Successful ---")
    else:
        print("--- Test Failed ---")