from .base_agent import BaseAgent
from utils import calculate_elo_update
import random
import copy

class RankingAgent(BaseAgent):
    """Agent for ranking ideas using Elo system"""
    
    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
        self.initial_elo = 1200
    
    def rank_ideas(self, ideas, experiment_content, research_goal, num_matches=10):
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
            # Select two different ideas for comparison
            idx1, idx2 = random.sample(range(len(ranked_ideas)), 2)
            idea1, idea2 = ranked_ideas[idx1], ranked_ideas[idx2]
            
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
        You are an expert evaluator tasked with comparing two research ideas for a machine learning experiment.
        
        Research Goal: {research_goal}
        
        Here is the experiment code:
        
        ```python
        {experiment_content}
        ```
        
        Idea 1:
        Name: {idea1["Name"]}
        Title: {idea1["Title"]}
        Experiment Description: {idea1["Experiment"]}
        Interestingness: {idea1["Interestingness"]}
        Feasibility: {idea1["Feasibility"]}
        Novelty: {idea1["Novelty"]}
        
        Idea 2:
        Name: {idea2["Name"]}
        Title: {idea2["Title"]}
        Experiment Description: {idea2["Experiment"]}
        Interestingness: {idea2["Interestingness"]}
        Feasibility: {idea2["Feasibility"]}
        Novelty: {idea2["Novelty"]}
        
        Please compare these two ideas and determine which one is superior in terms of:
        1. Alignment with the research goal
        2. Novelty and creativity
        3. Technical feasibility
        4. Potential impact
        
        For each criterion, provide a brief comparison. Then conclude with your final judgment on which idea is better overall.
        
        End your response with one of these statements:
        - "Winner: Idea 1" if the first idea is better
        - "Winner: Idea 2" if the second idea is better
        """
        
        return prompt
    
    def process_response(self, response):
        """Process the LLM response to determine the winner"""
        try:
            # Look for explicit winner statement
            if "Winner: Idea 1" in response:
                return 1
            elif "Winner: Idea 2" in response:
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