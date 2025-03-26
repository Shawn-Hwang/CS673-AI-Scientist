from .base_agent import BaseAgent

class ReflectionAgent(BaseAgent):
    """Agent for reviewing and evaluating ideas"""
    
    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
    
    def review_ideas(self, ideas, experiment_content, research_goal):
        """Review a list of ideas and filter/improve them"""
        reviewed_ideas = []
        
        for idea in ideas:
            review_result = self.review_idea(idea, experiment_content, research_goal)
            
            if review_result["accepted"]:
                reviewed_ideas.append(review_result["updated_idea"])
            else:
                print(f"Idea '{idea['Name']}' was rejected during review")
        
        return reviewed_ideas
    
    def review_idea(self, idea, experiment_content, research_goal):
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
    
    def process_response(self, response, original_idea):
        """Process the LLM response into a structured review"""
        try:
            # Try to extract JSON from the response
            review_json = self.extract_json(response)
            
            if review_json:
                # Get all metrics from the review
                correctness = review_json.get("Correctness", {}).get("Score", 5)
                quality = review_json.get("Quality", {}).get("Score", 5)
                novelty = review_json.get("Novelty", {}).get("Score", 5)
                feasibility = review_json.get("Feasibility", {}).get("Score", 5)
                
                # Get recommendation
                recommendation = review_json.get("Recommendation", "Accept")
                accepted = recommendation != "Reject"
                
                # Prepare feedback
                feedback = {
                    "CorrectnessFeedback": review_json.get("Correctness", {}).get("Justification", ""),
                    "QualityFeedback": review_json.get("Quality", {}).get("Justification", ""),
                    "NoveltyFeedback": review_json.get("Novelty", {}).get("Justification", ""),
                    "FeasibilityFeedback": review_json.get("Feasibility", {}).get("Justification", ""),
                    "ImprovementSuggestions": review_json.get("Improvement Suggestions", ""),
                    "OverallAssessment": review_json.get("Overall Assessment", "")
                }
                
                # Update the idea with reviewed scores
                updated_idea = original_idea.copy()
                updated_idea["Interestingness"] = max(original_idea.get("Interestingness", 5), quality)
                updated_idea["Feasibility"] = max(original_idea.get("Feasibility", 5), feasibility)
                updated_idea["Novelty"] = max(original_idea.get("Novelty", 5), novelty)
                
                # Add review notes
                review_notes = (
                    f"Review Notes:\n"
                    f"Correctness: {correctness}/10 - {feedback['CorrectnessFeedback']}\n"
                    f"Quality: {quality}/10 - {feedback['QualityFeedback']}\n"
                    f"Novelty: {novelty}/10 - {feedback['NoveltyFeedback']}\n"
                    f"Feasibility: {feasibility}/10 - {feedback['FeasibilityFeedback']}\n"
                    f"Improvement Suggestions: {feedback['ImprovementSuggestions']}\n"
                    f"Overall Assessment: {feedback['OverallAssessment']}"
                )
                
                updated_idea["Notes"] = updated_idea.get("Notes", "") + "\n\n" + review_notes
                
                return {
                    "accepted": accepted,
                    "recommendation": recommendation,
                    "feedback": feedback,
                    "updated_idea": updated_idea
                }
            else:
                # Fallback if no JSON could be extracted
                print("Warning: Could not extract JSON from Reflection Agent response")
                return {
                    "accepted": True,  # By default, accept the idea to allow pipeline to continue
                    "recommendation": "Accept",
                    "feedback": {
                        "OverallAssessment": "Could not parse structured review from response"
                    },
                    "updated_idea": original_idea
                }
        except Exception as e:
            print(f"Error processing Reflection Agent response: {e}")
            # Return a fallback review
            return {
                "accepted": True,  # By default, accept the idea to allow pipeline to continue
                "recommendation": "Accept",
                "feedback": {
                    "OverallAssessment": "Error processing review, default feedback provided",
                },
                "updated_idea": original_idea
            }