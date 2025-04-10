from .base_agent import BaseAgent

class DebateAgent(BaseAgent):
    """Agent for managing debates on generated ideas"""

    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
    
    def debate_ideas(self, ideas, experiment_content, research_goal):
        """Review a list of ideas and filter/improve them"""
        reviewed_ideas = []
        
        for idea in ideas:
            review_result = self.review_idea(idea, experiment_content, research_goal)
            
            if review_result["accepted"]:
                reviewed_ideas.append(review_result["updated_idea"])
            else:
                print(f"Idea '{idea['Name']}' was rejected during review")
        
        return reviewed_ideas
    
    def debate_idea(self, idea, experiment_content, research_goal):
        """Review a single idea and provide feedback"""
        # Prepare prompt
        prompt = self.prepare_prompt(
            idea=idea,
            experiment_content=experiment_content,
            research_goal=research_goal
        )
        
        # Call LLM
        response = self.call_llm(prompt)
        
        # Process response
        review_result = self.process_response(response, idea)
        
        return review_result
    
    def prepare_prompt(self, idea, experiment_content, research_goal):
        """Prepare prompt for idea review"""
        prompt = f"""
        You are an expert scientific reviewer evaluating a proposed idea for improving a machine learning experiment.
        
        Research Goal: {research_goal}
        
        Here is the experiment code:
        
        ```python
        {experiment_content}
        ```
        
        Here is the idea to review:
        
        Name: {idea["Name"]}
        Title: {idea["Title"]}
        Experiment Description: {idea["Experiment"]}
        
        Please conduct a critical review of this idea, focusing on:
        
        1. Correctness: Is the idea technically sound and free of errors?
        2. Quality: Is the idea well-developed, clear, and likely to lead to improvements?
        3. Novelty: Is the idea original and does it provide new insights?
        4. Feasibility: Is the idea practical to implement with reasonable resources?
        
        For each aspect, provide a score (1-10) and justification.
        Then make an overall recommendation: Accept, Revise, or Reject.
        
        If recommending Revise or Accept, suggest improvements or refinements to strengthen the idea.
        
        Format your response as a JSON object with the following structure:
        {{
            "Correctness": {{
                "Score": score,
                "Justification": "your reasoning"
            }},
            "Quality": {{
                "Score": score,
                "Justification": "your reasoning"
            }},
            "Novelty": {{
                "Score": score,
                "Justification": "your reasoning"
            }},
            "Feasibility": {{
                "Score": score,
                "Justification": "your reasoning"
            }},
            "Recommendation": "Accept/Revise/Reject",
            "Improvement Suggestions": "your suggestions if any",
            "Overall Assessment": "summary of your evaluation"
        }}
        """
        
        return prompt
    
    def process_response(self, response):
        """Process the LLM response into structured ideas"""
        try:
            # Try to extract JSON from the response
            ideas_json = self.extract_json(response)
            
            if ideas_json:
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