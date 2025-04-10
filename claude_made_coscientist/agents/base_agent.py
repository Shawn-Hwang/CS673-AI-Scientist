from utils import call_llm, extract_json_from_text

class BaseAgent:
    """Base class for all co-scientist agents"""
    
    def __init__(self, use_genai=True, model="gemini-2.0-flash", persona=None):
        self.context = {}
        self.use_genai = use_genai
        self.model = model
        self.system_prompt = "You are an AI scientist specializing in machine learning research."
    
    def prepare_prompt(self, **kwargs):
        """Prepare a prompt for the LLM"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def process_response(self, response, **kwargs):
        """Process the response from the LLM"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def call_llm(self, prompt):
        """Call the LLM with the given prompt"""
        print(f"Calling LLM for {self.__class__.__name__} with prompt of length {len(prompt)}")
        return call_llm(prompt, model=self.model, use_genai=self.use_genai, system_prompt=self.system_prompt)
    
    def extract_json(self, text):
        """Extract JSON from text response"""
        return extract_json_from_text(text)