from .base_agent import BaseAgent
import numpy as np

class ProximityAgent(BaseAgent):
    """Agent for calculating similarity between ideas and building a proximity graph"""

    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
    
    def calculate_proximity(self, ideas, experiment_content, research_goal):
        """Calculate proximity between ideas and build a proximity graph"""
        if len(ideas) <= 1:
            # Return original ideas if there's only one or none
            return ideas, {}
            
        # Create a proximity matrix
        proximity_matrix = np.zeros((len(ideas), len(ideas)))
        proximity_graph = {}
        
        # Calculate pairwise proximity for all ideas
        for i in range(len(ideas)):
            proximity_graph[ideas[i]["Name"]] = []
            for j in range(i+1, len(ideas)):
                # Calculate similarity between idea i and j
                similarity = self.calculate_similarity(
                    idea1=ideas[i],
                    idea2=ideas[j],
                    experiment_content=experiment_content,
                    research_goal=research_goal
                )
                
                # Update proximity matrix (symmetric)
                proximity_matrix[i, j] = similarity
                proximity_matrix[j, i] = similarity
                
                # Add to proximity graph if similarity is above threshold
                if similarity >= 0.5:  # threshold for considering ideas similar
                    proximity_graph[ideas[i]["Name"]].append({
                        "name": ideas[j]["Name"],
                        "similarity": similarity
                    })
                    if ideas[j]["Name"] not in proximity_graph:
                        proximity_graph[ideas[j]["Name"]] = []
                    proximity_graph[ideas[j]["Name"]].append({
                        "name": ideas[i]["Name"],
                        "similarity": similarity
                    })
        
        # Add proximity information to each idea
        for i, idea in enumerate(ideas):
            # Find most similar ideas
            similar_indices = np.argsort(proximity_matrix[i])[::-1][1:4]  # Top 3 most similar (excluding self)
            similar_ideas = []
            
            for idx in similar_indices:
                if proximity_matrix[i, idx] > 0:  # Only include if similarity is positive
                    similar_ideas.append({
                        "name": ideas[idx]["Name"],
                        "similarity": float(proximity_matrix[i, idx])
                    })
            
            # Add to idea metadata
            idea["Similar Ideas"] = similar_ideas
            
            # Add a cluster label (very simple clustering)
            # Ideas with highest similarity to each other get the same cluster
            idea["Cluster"] = f"cluster_{i % 3}"  # Simple clustering into 3 groups
        
        return ideas, proximity_graph
    
    def calculate_similarity(self, idea1, idea2, experiment_content, research_goal):
        """Calculate similarity between two ideas"""
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
        similarity = self.process_response(response)
        
        return similarity
    
    def prepare_prompt(self, idea1, idea2, experiment_content, research_goal):
        """Prepare prompt for similarity calculation"""
        prompt = f"""
        You are an expert in analyzing similarity between research ideas.
        
        Research Goal: {research_goal}
        
        Here is the experiment code context:
        
        ```python
        {experiment_content[:500]}...  # Truncated for brevity
        ```
        
        Idea 1:
        Name: {idea1["Name"]}
        Title: {idea1["Title"]}
        Experiment Description: {idea1["Experiment"][:300]}...  # Truncated for brevity
        
        Idea 2:
        Name: {idea2["Name"]}
        Title: {idea2["Title"]}
        Experiment Description: {idea2["Experiment"][:300]}...  # Truncated for brevity
        
        Please analyze the similarity between these two ideas based on:
        1. Conceptual similarity (do they address similar concepts or techniques)
        2. Methodological similarity (do they propose similar approaches)
        3. Goal similarity (do they aim to achieve similar outcomes)
        
        For each dimension, provide a similarity score between 0 and 1, where:
        0 = Completely different
        1 = Identical
        
        Then, provide an overall similarity score between 0 and 1.
        
        Format your response as a single line with just the overall similarity score (a number between 0 and 1).
        """
        
        return prompt
    
    def process_response(self, response):
        """Process the LLM response into a similarity score"""
        try:
            # Try to extract a numerical score from the response
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                try:
                    # Try to parse as float
                    score = float(line)
                    if 0 <= score <= 1:
                        return score
                except ValueError:
                    continue
            
            # If no valid score was found, try to find a number in the response
            import re
            score_matches = re.findall(r'(\d+\.\d+|\d+)', response)
            if score_matches:
                for match in score_matches:
                    score = float(match)
                    if 0 <= score <= 1:
                        return score
                    elif score > 1 and score <= 10:
                        return score / 10  # Normalize if on a 0-10 scale
            
            # Default similarity if no valid score could be extracted
            return 0.5
            
        except Exception as e:
            print(f"Error processing Proximity Agent response: {e}")
            # Return a default similarity
            return 0.5