import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig


load_dotenv()

def generate_response(prompt, model="gemini-2.0-flash", temperature=0.75, max_tokens=5000, system_message=None):
    """
    Generates a text response from a language model.

    Args:
        prompt (str): The prompt to send to the language model.
        model (str): The name of the language model to use.
        temperature (float): Controls randomness (higher = more random).
        max_tokens (int): The maximum number of tokens in the response.
        system_message (str): System instructions for the model.
                              e.g. "You are a helpful coding assistant."
    Returns:
        str: The generated text response, or None if an error occurred.
    """
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],)
        
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
                system_instruction=system_message,
            ),
        )
        content = response.text
        print(content)
        return content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def summarize_text(text, model="gemini-2.0-flash", max_tokens=300):
    """Summarizes a given text using a language model."""
    prompt = f"Summarize the following text:\n{text}\n\nSummary:"
    return generate_response(prompt, model=model, max_tokens=max_tokens)