from .base_agent import BaseAgent
import re
import copy

class DebateAgent(BaseAgent):
    """Agent for simulating scientific debates among personas to improve ideas"""
    
    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
        self.system_prompt = "You are facilitating a scientific debate among different research personas to improve machine learning research ideas."
    
    def debate_ideas(self, ideas, experiment_content, research_goal, temperature):
        """Simulate debates for each idea and return improved ideas"""
        debated_ideas = []
        
        for idea in ideas:
            # Extract persona that generated this idea
            original_persona = idea.get("Persona", "An expert AI researcher")
            
            # Extract personas from other ideas to serve as critics
            other_personas = [other_idea.get("Persona", "An expert AI researcher") 
                             for other_idea in ideas if other_idea["Name"] != idea["Name"]]
            
            # If there are no other personas, create generic critics
            if not other_personas or len(other_personas) < 2:
                generic_critics = ["A critical AI researcher with expertise in model efficiency", 
                                  "A skeptical AI researcher who focuses on experimental rigor"]
                other_personas = (other_personas + generic_critics)[:2]
            
            print(f"Debating idea: {idea['Name']}")
            print(f"Original persona: {original_persona}")
            print(f"Critics: {other_personas[:2]}")
            
            # Run debate on this idea
            debated_idea = self.debate_idea(
                idea=idea,
                original_persona=original_persona,
                critic_personas=other_personas[:2],  # Limit to 2 critics for brevity
                experiment_content=experiment_content,
                research_goal=research_goal,
                temperature=temperature
            )
            
            debated_ideas.append(debated_idea)
        
        return debated_ideas
    
    def debate_idea(self, idea, original_persona, critic_personas, experiment_content, research_goal, temperature):
        """Run a debate for a single idea and return the improved idea"""
        # Prepare prompt for the debate
        prompt = self.prepare_prompt(
            idea=idea,
            original_persona=original_persona,
            critic_personas=critic_personas,
            experiment_content=experiment_content,
            research_goal=research_goal
        )
        
        # Call LLM for the debate
        response = self.call_llm(prompt, temperature=temperature)
        
        # Process response
        debated_idea = self.process_response(response, idea)
        
        return debated_idea
    
    def prepare_prompt(self, idea, original_persona, critic_personas, experiment_content, research_goal):
        """Prepare prompt for idea debate"""
        # Format the idea information
        idea_info = f"""
        Name: {idea["Name"]}
        Title: {idea["Title"]}
        Experiment Description: {idea["Experiment"]}
        Interestingness: {idea["Interestingness"]}/10
        Feasibility: {idea["Feasibility"]}/10
        Novelty: {idea["Novelty"]}/10
        """
        
        # Create the debate prompt
        prompt = f"""
        You will simulate a scientific debate among different personas about a research idea for improving a machine learning experiment.
        
        Research Goal: {research_goal}
        
        Here is the experiment code context (truncated for brevity):
        
        ```python
        {experiment_content}...
        ```
        
        Here is the idea to debate:
        
        {idea_info}
        
        This idea was originally proposed by a persona with the following description:
        "{original_persona}"
        
        Simulate a debate with the following structure:
        
        1. First, have the original persona promote their idea, highlighting its strengths and potential impact.
        
        2. Then, have each of the following critic personas give their perspective, pointing out potential weaknesses or areas for improvement:
        """
        
        # Add each critic persona to the prompt
        for i, persona in enumerate(critic_personas):
            prompt += f"\n   Critic {i+1}: {persona}"
        
        prompt += f"""
        
        3. Finally, synthesize the perspectives from the debate and create an improved version of the idea that addresses the critiques while maintaining its core strengths.
        While you are improving the idea, please ensure that the idea is feasible to investigate given the provided python code.
        
        Format your response as follows:
        
        Original Persona's Promotion:
        [The original persona's promotion of the idea]
        
        Critic 1's Perspective:
        [First critic's perspective]
        
        Critic 2's Perspective:
        [Second critic's perspective]
        
        Debate Summary:
        [Summary of the key points from the debate]
        
        Improved Idea:
        ```json
        {{
            "Name": "improved_idea_name",
            "Title": "Improved Research Paper Style Title",
            "Experiment": "Improved detailed explanation and implementation steps",
            "Interestingness": improved_score,
            "Feasibility": improved_score,
            "Novelty": improved_score,
            "Notes": "Additional thoughts including insights from the debate",
            "Persona": "{original_persona}"
        }}
        ```
        
        Ensure the improved idea maintains its core concept but addresses the valid criticisms and incorporates the valuable suggestions from the debate.
        """
        
        return prompt
    
    def process_response(self, response, original_idea):
        """Process the LLM response into an improved idea"""
        try:
            # Extract debate sections
            promotion_match = re.search(r"Original Persona's Promotion:(.*?)Critic 1's Perspective:", response, re.DOTALL)
            promotion = promotion_match.group(1).strip() if promotion_match else "Promotion not extracted."
            
            critic1_match = re.search(r"Critic 1's Perspective:(.*?)Critic 2's Perspective:", response, re.DOTALL)
            critic1 = critic1_match.group(1).strip() if critic1_match else "Critic 1 not extracted."
            
            critic2_match = re.search(r"Critic 2's Perspective:(.*?)Debate Summary:", response, re.DOTALL)
            critic2 = critic2_match.group(1).strip() if critic2_match else "Critic 2 not extracted."
            
            summary_match = re.search(r"Debate Summary:(.*?)Improved Idea:", response, re.DOTALL)
            debate_summary = summary_match.group(1).strip() if summary_match else "Summary not extracted."
            
            # Try to extract JSON from the response
            idea_json = self.extract_json(response)
            
            if idea_json:
                # Create a copy of the original idea as the base
                debated_idea = copy.deepcopy(original_idea)
                
                # Update with the improved idea fields
                debated_idea["Name"] = idea_json.get("Name", debated_idea["Name"])
                debated_idea["Title"] = idea_json.get("Title", debated_idea["Title"])
                debated_idea["Experiment"] = idea_json.get("Experiment", debated_idea["Experiment"])
                debated_idea["Interestingness"] = idea_json.get("Interestingness", debated_idea["Interestingness"])
                debated_idea["Feasibility"] = idea_json.get("Feasibility", debated_idea["Feasibility"])
                debated_idea["Novelty"] = idea_json.get("Novelty", debated_idea["Novelty"])
                
                # Preserve original persona
                debated_idea["Persona"] = idea_json.get("Persona", debated_idea.get("Persona", "An expert AI researcher"))
                
                # Compile the debate into a cohesive narrative
                debate_notes = f"""
                ## Debate on this Idea
                
                ### Original Persona's Promotion
                {promotion}
                
                ### Critic 1's Perspective
                {critic1}
                
                ### Critic 2's Perspective
                {critic2}
                
                ### Debate Summary
                {debate_summary}
                """
                
                # Add debate notes to the original notes
                original_notes = debated_idea.get("Notes", "")
                debated_idea["Notes"] = original_notes + "\n\n" + debate_notes
                
                # Ensure scores are within range
                debated_idea["Interestingness"] = max(1, min(10, debated_idea["Interestingness"]))
                debated_idea["Feasibility"] = max(1, min(10, debated_idea["Feasibility"]))
                debated_idea["Novelty"] = max(1, min(10, debated_idea["Novelty"]))
                
                # Mark as debated
                debated_idea["Debated"] = True
                
                return debated_idea
            else:
                # Fallback if no JSON could be extracted
                print("Warning: Could not extract JSON from Debate Agent response")
                
                # Compile the debate into a cohesive narrative
                debate_notes = f"""
                ## Debate on this Idea (No improved idea extracted)
                
                ### Original Persona's Promotion
                {promotion}
                
                ### Critic 1's Perspective
                {critic1}
                
                ### Critic 2's Perspective
                {critic2}
                
                ### Debate Summary
                {debate_summary}
                """
                
                # Add the debate summary to notes and return a copy of the original idea
                debated_idea = copy.deepcopy(original_idea)
                debated_idea["Notes"] = debated_idea.get("Notes", "") + "\n\n" + debate_notes
                debated_idea["Debated"] = True
                return debated_idea
                
        except Exception as e:
            print(f"Error processing Debate Agent response: {e}")
            # Return a copy of the original idea as fallback with a note
            debated_idea = copy.deepcopy(original_idea)
            debated_idea["Notes"] = debated_idea.get("Notes", "") + "\n\nNote: Debate process encountered an error, returning original idea."
            debated_idea["Debated"] = True
            return debated_idea