from .base_agent import BaseAgent
from utils import calculate_elo_update
import random
import copy
import numpy as np

class RankingAgent(BaseAgent):
    """Agent for ranking ideas using Elo system"""
    
    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
        self.initial_elo = 1200
    
    def rank_ideas(self, ideas, proximity_matrix, experiment_content, research_goal, num_matches=10):
        """Rank ideas using pairwise comparisons and an Elo system"""
        # Create deep copies to avoid modifying originals
        ranked_ideas = copy.deepcopy(ideas)
        
        # Initialize ELO ratings if not present
        for idea in ranked_ideas:
            if "ELO rating" not in idea:
                idea["ELO rating"] = self.initial_elo
        
        # If there are fewer than 2 ideas, return them as is
        if len(ranked_ideas) < 2:
            return ranked_ideas
        
        # Determine number of matches to run
        actual_matches = min(num_matches, len(ranked_ideas) * (len(ranked_ideas) - 1) // 2)
        print(f"Running {actual_matches} tournament matches")
        
        # Run tournament matches
        for _ in range(actual_matches):
            # # Select two different ideas for comparison
            # idx1, idx2 = random.sample(range(len(ranked_ideas)), 2)
            # idea1, idea2 = ranked_ideas[idx1], ranked_ideas[idx2]

            # prioritize new ideas and high ranking ideas for matches
            match_probs = np.array([idea["ELO rating"] for idea in ranked_ideas])
            idea1_idx = np.random.choice(np.arange(len(ranked_ideas)), p=match_probs/np.sum(match_probs))
            idea1 = ranked_ideas[idea1_idx]

            # choose the second idea based on proximity to the first idea
            match_probs = proximity_matrix[idea1_idx]
            idea2_idx = np.random.choice(np.arange(len(ranked_ideas)), p=match_probs/np.sum(match_probs))
            idea2 = ranked_ideas[idea2_idx]

            # Compare the ideas
            winner_idx = self.compare_ideas(idea1, idea2, experiment_content, research_goal)
            
            # Update ELO ratings
            if winner_idx == 1:
                idea1["ELO rating"], idea2["ELO rating"] = calculate_elo_update(
                    idea1["ELO rating"], idea2["ELO rating"], 1)
                print(f"Match: '{idea1['Name']}' ({idea1['ELO rating']}) wins against '{idea2['Name']}' ({idea2['ELO rating']})")
            else:
                idea2["ELO rating"], idea1["ELO rating"] = calculate_elo_update(
                    idea2["ELO rating"], idea1["ELO rating"], 1)
                print(f"Match: '{idea2['Name']}' ({idea2['ELO rating']}) wins against '{idea1['Name']}' ({idea1['ELO rating']})")
        
        # Sort ideas by ELO rating
        ranked_ideas.sort(key=lambda x: x["ELO rating"], reverse=True)
        
        return ranked_ideas
    
    def compare_ideas(self, idea1, idea2, experiment_content, research_goal):
        """Compare two ideas and determine which is better"""
        # Prepare prompt
        prompt = self.prepare_prompt(
            idea1=idea1,
            idea2=idea2,
            experiment_content=experiment_content,
            research_goal=research_goal
        )
        
        # Call LLM
        response = self.call_llm(prompt)
        
        # Process response
        winner_idx = self.process_response(response)
        
        return winner_idx
    
    def prepare_prompt(self, idea1, idea2, experiment_content, research_goal):
        """Prepare prompt for idea comparison"""
        prompt = f"""
        You are an expert in comparative analysis, simulating a panel of machine learning domain experts
        engaged in a structured discussion to evaluate two competing ideas.
        The objective is to rigorously determine which idea is superior based on
        a predefined set of attributes and criteria.
        The experts possess no pre-existing biases toward either idea and are solely
        focused on identifying the optimal choice, given that only one can be implemented.
        
        Research Goal: {research_goal}
        
        Here is the experiment code:
        
        ```python
        {experiment_content}
        ```
        
        Idea 1:
        Name: {idea1["Name"]}
        Title: {idea1["Title"]}
        Experiment Description: {idea1["Experiment"]}
        
        Idea 2:
        Name: {idea2["Name"]}
        Title: {idea2["Title"]}
        Experiment Description: {idea2["Experiment"]}
        
        Please compare these two ideas and determine which one is superior in terms of:
        1. Alignment with the research goal
        2. Novelty and creativity
        3. Technical feasibility
        4. Potential impact
        
        Debate procedure:
        The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.
        Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.
        Subsequent turns:
        * Pose clarifying questions to address any ambiguities or uncertainties.
        * Critically evaluate each idea in relation to the stated Goal and Criteria.
        This evaluation should consider aspects such as:
        - Potential for performance increase.
        - Utility and practical applicability.
        - Sufficiency of detail and specificity.
        - Novelty and originality.
        - Parameter efficiency and computational complexity.
        - Potential for generalization and scalability.
        * Identify and articulate any weaknesses, limitations, or potential flaws in either idea.

        Termination and judgment:
        Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns)
        and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment.
        This judgment should succinctly state the rationale for the selection.
        Then, indicate the superior idea by writing the phrase "better idea: ",
        followed by "1" (for idea 1) or "2" (for idea 2).
        """
        
        return prompt
    
    def process_response(self, response):
        """Process the LLM response to determine the winner"""
        try:
            # Look for explicit winner statement
            if "better idea: 1" in response.lower():
                return 1
            elif "better idea: 2" in response.lower():
                return 2
            
            # If no explicit winner found, analyze the response
            response_lower = response.lower()
            
            positive_words = ["better", "superior", "stronger", "higher", "more promising", 
                             "more novel", "more interesting", "more feasible"]
            
            # Count positive associations
            idea1_positive = sum([response_lower.count(f"{word} idea 1") + 
                                 response_lower.count(f"idea 1 is {word}") for word in positive_words])
            idea2_positive = sum([response_lower.count(f"{word} idea 2") + 
                                 response_lower.count(f"idea 2 is {word}") for word in positive_words])
            
            return 1 if idea1_positive >= idea2_positive else 2
            
        except Exception as e:
            print(f"Error processing Ranking Agent response: {e}")
            # Return a random winner as fallback
            return random.choice([1, 2])