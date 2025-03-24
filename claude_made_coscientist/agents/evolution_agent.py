from .base_agent import BaseAgent
import copy

class EvolutionAgent(BaseAgent):
    """Agent for evolving and improving ideas"""

    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
    
    def evolve_ideas(self, ideas, experiment_content, research_goal, num_to_evolve=3):
        """Evolve a set of top ideas"""
        if not ideas:
            return []
            
        # Select top ideas to evolve
        top_ideas = ideas[:min(num_to_evolve, len(ideas))]
        evolved_ideas = []
        
        for idea in top_ideas:
            evolved_idea = self.evolve_idea(idea, experiment_content, research_goal)
            if evolved_idea:
                evolved_ideas.append(evolved_idea)
                
        return evolved_ideas
    
    def evolve_idea(self, idea, experiment_content, research_goal):
        """Evolve and improve a single idea"""
        # Prepare prompt
        prompt = self.prepare_prompt(
            idea=idea,
            experiment_content=experiment_content,
            research_goal=research_goal
        )
        
        # Call LLM
        response = self.call_llm(prompt)
        
        # Process response
        evolved_idea = self.process_response(response, idea)
        
        return evolved_idea
    
    def prepare_prompt(self, idea, experiment_content, research_goal):
        """Prepare prompt for idea evolution"""
        prompt = f"""
        You are an expert in machine learning research tasked with improving an existing research idea.
        
        Research Goal: {research_goal}
        
        Here is the experiment code:
        
        ```python
        {experiment_content}
        ```
        
        Here is the idea to improve:
        
        Name: {idea["Name"]}
        Title: {idea["Title"]}
        Experiment Description: {idea["Experiment"]}
        Interestingness: {idea["Interestingness"]}
        Feasibility: {idea["Feasibility"]}
        Novelty: {idea["Novelty"]}
        Notes: {idea["Notes"]}
        
        Please improve this idea by:
        1. Enhancing its technical aspects
        2. Making it more practical and feasible
        3. Increasing its novelty and interestingness
        4. Addressing any limitations or weaknesses
        
        Create an evolved version of this idea that maintains its core concept but improves upon it.
        
        Format your response as a JSON object with the following structure:
        {{
            "Name": "evolved_ideaName",
            "Title": "Evolved Research Paper Title",
            "Experiment": "Improved experiment description",
            "Interestingness": improved_score,
            "Feasibility": improved_score,
            "Novelty": improved_score,
            "Notes": "Notes on the improvements made"
        }}
        
        Ensure the evolved idea is significantly better than the original while remaining true to its core concept.
        """
        
        return prompt
    
    def process_response(self, response, original_idea):
        """Process the LLM response into an evolved idea"""
        try:
            # Try to extract JSON from the response
            evolved_json = self.extract_json(response)
            
            if evolved_json:
                # Create a new idea object with evolved properties
                evolved_idea = {
                    "Name": evolved_json.get("Name", f"evolved_{original_idea['Name']}"),
                    "Title": evolved_json.get("Title", f"Evolved: {original_idea['Title']}"),
                    "Experiment": evolved_json.get("Experiment", original_idea["Experiment"]),
                    "Interestingness": evolved_json.get("Interestingness", original_idea["Interestingness"]),
                    "Feasibility": evolved_json.get("Feasibility", original_idea["Feasibility"]),
                    "Novelty": evolved_json.get("Novelty", original_idea["Novelty"]),
                    "Notes": evolved_json.get("Notes", "") + f"\n\nEvolved from original idea: {original_idea['Name']}",
                    "ELO rating": 1200  # Reset ELO for the evolved idea
                }
                
                # Ensure scores are within range
                evolved_idea["Interestingness"] = max(1, min(10, evolved_idea["Interestingness"]))
                evolved_idea["Feasibility"] = max(1, min(10, evolved_idea["Feasibility"]))
                evolved_idea["Novelty"] = max(1, min(10, evolved_idea["Novelty"]))
                
                return evolved_idea
            else:
                # Fallback if no JSON could be extracted
                print("Warning: Could not extract JSON from Evolution Agent response")
                fallback_idea = copy.deepcopy(original_idea)
                fallback_idea["Name"] = f"evolved_{original_idea['Name']}"
                fallback_idea["Title"] = f"Evolved: {original_idea['Title']}"
                fallback_idea["Notes"] += "\n\nEvolution notes: " + response[:200] + "..."
                fallback_idea["ELO rating"] = 1200
                return fallback_idea
                
        except Exception as e:
            print(f"Error processing Evolution Agent response: {e}")
            # Return a slightly modified copy of the original as fallback
            fallback_idea = copy.deepcopy(original_idea)
            fallback_idea["Name"] = f"evolved_{original_idea['Name']}"
            fallback_idea["Notes"] += "\nNote: Evolution with minimal changes due to processing error."
            fallback_idea["ELO rating"] = 1200
            return fallback_idea