from .base_agent import BaseAgent
import random
import json
import polars as pl
from datasets import load_dataset

class GenerationAgent(BaseAgent):
    """Agent for generating initial ideas"""

    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)

    def sample_personas(self, num_personas=3):
        """Sample personas for idea generation"""
        # path = '/home/huang717/nobackup/archive/hf_datasets/PersonaHub/ElitePersona/elite_personas.part1.jsonl'
        # df = pl.read_ndjson(path)

        path = '/home/huang717/CS673-AI-Scientist/claude_made_coscientist/agents/sampled_elite_personas.csv'
        df = pl.read_csv(path)

        sampled_personas = df.sample(n=3).to_pandas()['persona'].to_list()
        
        return sampled_personas
    
    def generate_ideas(self, experiment_content, research_goal, skip_lit_review=False, num_ideas=5):
        """Generate initial ideas based on the experiment.py content"""

        # Prepare prompt
        prompt = self.prepare_prompt(
            experiment_content=experiment_content,
            research_goal=research_goal,
            skip_lit_review=skip_lit_review,
            num_ideas=num_ideas,
        )
        
        # Call LLM
        response = self.call_llm(prompt)
        
        # Process response
        ideas = self.process_response(response)
        
        return ideas
    
    def generate_ideas_with_personas(self, experiment_content, research_goal, skip_lit_review=False, num_ideas=5):
        """Generate initial ideas based on the experiment.py content with personas"""
        idea_str_archive = []
        ideas = []

        # Sample personas
        personas = self.sample_personas(num_personas=num_ideas)

        for i in range(num_ideas):
            print()
            print(f"Generating idea {i + 1}/{num_ideas}")

            try:
                prev_ideas_string = "\n\n".join(idea_str_archive)
                persona_description = personas[i]

                # Prepare prompt
                prompt = self.prepare_persona_prompt(
                    experiment_content=experiment_content,
                    research_goal=research_goal,
                    skip_lit_review=skip_lit_review,
                    persona=persona_description,
                    prev_ideas_string=prev_ideas_string,
                )

                # Call LLM
                response = self.call_llm(prompt,temperature=1.8)
                # print(f"Response: {response}")
                
                # Process response
                idea = self.process_response(response)[0]
                idea['Persona'] = persona_description

                # print("****************************************************************")
                # print(f"Processed idea: {idea}, type: {type(idea)}")
                # print("****************************************************************")

                ideas.append(idea)

                idea_str_archive.append(json.dumps(idea))

            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue
        
        return ideas
    
    def prepare_persona_prompt(self, experiment_content, research_goal, skip_lit_review, persona, prev_ideas_string):
        """Prepare prompt for idea generation"""
        if skip_lit_review:
            lit_review_text = "Skip literature review as requested."
        else:
            lit_review_text = "Conduct a brief literature review on relevant techniques and approaches that could improve this experiment."
        
        prompt = f"""
        You are an expert AI researcher tasked with generating novel ideas to improve a machine learning experiment.
        You are also an {persona}
        
        Research Goal: {research_goal}
        
        Here is the experiment code to analyze:
        
        ```python
        {experiment_content}
        ```
        
        {lit_review_text}

        Here are the ideas that you have already generated:

        '''
        {prev_ideas_string}
        '''
        
        Please use your domain knowledge to come up with a unique and innovative approach for improving or modifying this experiment. Make sure to make your idea distinct from the previous ones. 
        For your novel idea, please provide:
        
        1. A short name (lowercase, underscore-separated)
        2. A title in the style of a research paper
        3. A detailed explanation of the idea and how to implement it
        4. An interestingness score (1-10)
        5. A feasibility score (1-10)
        6. A novelty score (1-10)
        7. Additional notes or justification for the idea
        
        Format your response as a JSON list with the following structure for your idea:
        {{
            "Name": "idea_name",
            "Title": "Research Paper Style Title",
            "Experiment": "Detailed explanation and implementation steps",
            "Interestingness": score,
            "Feasibility": score,
            "Novelty": score,
            "Notes": "Additional thoughts and justification"
        }}
        
        Ensure your idea is creative, well-justified, and technically sound. It should introduce meaningful
        improvements to the experiment's architecture, training process, evaluation methodology, or objectives.
        """
        
        return prompt
    
    
    def prepare_prompt(self, experiment_content, research_goal, skip_lit_review, num_ideas):
        """Prepare prompt for idea generation"""
        if skip_lit_review:
            lit_review_text = "Skip literature review as requested."
        else:
            lit_review_text = "Conduct a brief literature review on relevant techniques and approaches that could improve this experiment."
        
        prompt = f"""
        You are an expert AI researcher tasked with generating novel ideas to improve a machine learning experiment.
        
        Research Goal: {research_goal}
        
        Here is the experiment code to analyze:
        
        ```python
        {experiment_content}
        ```
        
        {lit_review_text}
        
        Please generate {num_ideas} novel ideas for improving or modifying this experiment. For each idea, provide:
        
        1. A short name (lowercase, underscore-separated)
        2. A title in the style of a research paper
        3. A detailed explanation of the idea and how to implement it
        4. An interestingness score (1-10)
        5. A feasibility score (1-10)
        6. A novelty score (1-10)
        7. Additional notes or justification for the idea
        
        Format your response as a JSON list with the following structure for each idea:
        {{
            "Name": "idea_name",
            "Title": "Research Paper Style Title",
            "Experiment": "Detailed explanation and implementation steps",
            "Interestingness": score,
            "Feasibility": score,
            "Novelty": score,
            "Notes": "Additional thoughts and justification"
        }}
        
        Ensure your ideas are creative, well-justified, and technically sound. They should introduce meaningful
        improvements to the experiment's architecture, training process, evaluation methodology, or objectives.
        """
        
        return prompt
    
    def process_response(self, response):
        """Process the LLM response into structured ideas"""
        try:
            # Try to extract JSON from the response
            ideas_json = self.extract_json(response)
            
            
            if ideas_json:
                # Convert to list if a single idea was returned (not in a list)
                if isinstance(ideas_json, dict):
                    ideas_json = [ideas_json]

                # Ensure each idea has the required fields and initialize ELO rating
                for i, idea in enumerate(ideas_json):
                    if "Name" not in idea:
                        idea["Name"] = "unnamed_idea"
                    if "Title" not in idea:
                        idea["Title"] = "Untitled Idea"
                    if "Experiment" not in idea:
                        idea["Experiment"] = "No experiment description provided"
                    if "Interestingness" not in idea:
                        idea["Interestingness"] = 5
                    if "Feasibility" not in idea:
                        idea["Feasibility"] = 5
                    if "Novelty" not in idea:
                        idea["Novelty"] = 5
                    if "Notes" not in idea:
                        idea["Notes"] = "No additional notes provided"
                    if "Evolved" not in idea:
                        idea["Evolved"] = False
                    
                    # Add ELO rating
                    idea["ELO rating"] = 1200
                    
                    # Ensure scores are within range
                    idea["Interestingness"] = max(1, min(10, idea["Interestingness"]))
                    idea["Feasibility"] = max(1, min(10, idea["Feasibility"]))
                    idea["Novelty"] = max(1, min(10, idea["Novelty"]))
                    idea["Idea number"] = i + 1
                
                return ideas_json
            else:
                # Fallback if no JSON could be extracted
                print("Warning: Could not extract JSON from Generation Agent response")
                return [{
                    "Name": "parsed_idea",
                    "Title": "Idea Parsed from Non-JSON Response",
                    "Experiment": response[:500] + "...", # Truncate long responses
                    "Interestingness": 5,
                    "Feasibility": 5,
                    "Novelty": 5,
                    "ELO rating": 1200,
                    "Notes": "Generated from non-JSON response"
                }]
        except Exception as e:
            print(f"Error processing Generation Agent response: {e}")
            # Return a fallback idea
            return [{
                "Name": "fallback_idea",
                "Title": "Fallback Idea Due to Processing Error",
                "Experiment": "This is a fallback idea due to processing error",
                "Interestingness": 5,
                "Feasibility": 5,
                "Novelty": 5,
                "ELO rating": 1200,
                "Notes": "Generated as fallback due to response processing error"
            }]