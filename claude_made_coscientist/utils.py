import json
import os
import re
from google import genai
from google.genai.types import GenerateContentConfig

def setup_genai_api():
    """Initialize Google Generative AI API with key from environment"""
    if 'GEMINI_API_KEY' in os.environ:
        # No need to call configure anymore, we'll create clients directly
        return True
    else:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Using dummy responses for testing.")
        return False

def call_llm(prompt, model="gemini-2.0-flash", temperature=0.7, use_genai=True, system_prompt=None):
    """Call LLM with prompt and return response"""
    if not use_genai:
        # Dummy response for testing without API
        return f"Simulated LLM response for prompt: {prompt[:50]}..."
    
    try:
        # Create client
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        
        # If no system prompt is provided, use a default one
        if system_prompt is None:
            system_prompt = "You are an AI scientist specializing in machine learning research."
        
        # Format message
        contents = [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
        
        # Call API
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=8192,  # Reasonable default
                candidate_count=1,
                system_instruction=system_prompt,
            ),
        )
        
        return response.text
    except Exception as e:
        print(f"Error calling Google Generative AI API: {e}")
        return f"Error: {str(e)}"

# def extract_json_from_text(text):
#     """Extract JSON object from text response"""
#     try:
#         # First, try to find JSON between triple backticks
#         json_match = re.search(r"```json\n(.*?)```", text, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(1)
#             return json.loads(json_str)
        
#         # Second, try to find JSON between single backticks
#         json_match = re.search(r"`(.*?)`", text, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(1)
#             return json.loads(json_str)
        
#         # Third, try to find JSON-like structure without backticks
#         json_match = re.search(r"\{.*\}", text, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(0)
#             return json.loads(json_str)
        
#         # Try to parse the entire response as JSON
#         return json.loads(text)
#     except Exception as e:
#         print(f"Error extracting JSON from text: {e}")
#         return None

def extract_json_from_text(text):
    """Extract JSON object from text response with improved error handling"""
    try:
        # First, try to find JSON between triple backticks
        json_match = re.search(r"```(?:json)?\n([\s\S]*?)```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                # print("***********************************************************************************")
                # print(f"Extracted JSON: {json.loads(json_str)}")
                # print("***********************************************************************************")
                
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON within backticks: {e}")
                # Try to fix common JSON issues
                fixed_json_str = fix_json_string(json_str)
                return json.loads(fixed_json_str)
        
        # Second, try to find JSON-like structure without backticks
        # This looks for anything that might be a JSON object or array
        json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON without backticks: {e}")
                # Try to fix common JSON issues
                fixed_json_str = fix_json_string(json_str)
                return json.loads(fixed_json_str)
        
        print("No JSON structure found in response")
        return None
    except Exception as e:
        print(f"Error extracting JSON from text: {e}")
        return None

def fix_json_string(json_str):
    """Apply common fixes to JSON strings that might cause parsing errors"""
    # Replace unescaped quotes in strings
    fixed = re.sub(r'(?<!\\)"(.*?)(?<!\\)"(?=:)', r'"\1"', json_str)
    
    # Fix common issues with newlines in strings
    fixed = fixed.replace('\n', '\\n')
    
    # Handle potential trailing commas in arrays and objects
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    
    print(f"Attempted to fix JSON string, length: {len(fixed)}")
    return fixed

def calculate_elo_update(rating1, rating2, result, k=32):
    """Calculate Elo rating updates"""
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 - expected1
    
    new_rating1 = round(rating1 + k * (result - expected1))
    new_rating2 = round(rating2 + k * ((1-result) - expected2))
    
    return new_rating1, new_rating2