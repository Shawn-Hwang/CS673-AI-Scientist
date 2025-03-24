from .base_agent import BaseAgent
import copy

class MetaReviewAgent(BaseAgent):
    """Agent for final review and meta-analysis of ideas"""

    def __init__(self, use_genai=True, model="gemini-2.0-flash"):
        super().__init__(use_genai, model)
    
    def finalize_ideas(self, ideas, experiment_content, research_goal):
        """Finalize and provide meta-review for ideas"""
        if not ideas:
            return []
            
        # Prepare prompt
        prompt = self.prepare_prompt(
            ideas=ideas,
            experiment_content=experiment_content,
            research_goal=research_goal
        )
        
        # Call LLM
        response = self.call_llm(prompt)
        
        # Process response
        finalized_ideas = self.process_response(response, ideas)
        
        return finalized_ideas
    
    def prepare_prompt(self, ideas, experiment_content, research_goal):
        """Prepare prompt for meta-review"""
        # Create a condensed version of the ideas for the prompt
        ideas_text = "\n\n".join([
            f"Idea {i+1}:\nName: {idea['Name']}\nTitle: {idea['Title']}\nExperiment: {idea['Experiment'][:200]}...\nELO Rating: {idea['ELO rating']}"
            for i, idea in enumerate(ideas[:min(5, len(ideas))])  # Include at most 5 ideas to keep prompt size reasonable
        ])
        
        prompt = f"""
        You are a meta-reviewer tasked with analyzing a set of research ideas for a machine learning experiment.
        
        Research Goal: {research_goal}
        
        Here is the experiment code:
        
        ```python
        {experiment_content}
        ```
        
        Here are the top-ranked ideas:
        
        {ideas_text}
        
        Please provide a meta-analysis of these ideas, focusing on:
        
        1. Common themes and patterns across ideas
        2. Potential synergies or combinations of ideas
        3. Overall assessment of the ideas' quality and alignment with the research goal
        4. Recommendations for further exploration or improvement
        
        Your meta-review will be used to finalize the ideas before they are presented to researchers.
        
        Format your response as a JSON object with the following structure:
        {{
            "CommonThemes": "Analysis of common themes",
            "PotentialSynergies": "Description of potential combinations",
            "OverallAssessment": "General quality assessment",
            "Recommendations": "Specific recommendations for improvement",
            "IdeaFeedback": [
                {{
                    "IdeaName": "name_of_idea",
                    "Strengths": "Key strengths",
                    "Weaknesses": "Key weaknesses",
                    "SuggestedImprovements": "Specific improvement suggestions"
                }},
                ...
            ]
        }}
        """
        
        return prompt
    
    def process_response(self, response, original_ideas):
        """Process the LLM response and finalize ideas"""
        try:
            # Try to extract JSON from the response
            meta_review_json = self.extract_json(response)
            
            if meta_review_json:
                # Get meta-review content
                common_themes = meta_review_json.get("CommonThemes", "No common themes identified")
                potential_synergies = meta_review_json.get("PotentialSynergies", "No potential synergies identified")
                overall_assessment = meta_review_json.get("OverallAssessment", "No overall assessment provided")
                recommendations = meta_review_json.get("Recommendations", "No recommendations provided")
                
                # Get individual idea feedback
                idea_feedback = meta_review_json.get("IdeaFeedback", [])
                
                # Create a lookup for idea feedback by name
                feedback_by_name = {}
                for feedback in idea_feedback:
                    name = feedback.get("IdeaName", "")
                    if name:
                        feedback_by_name[name] = feedback
                
                # Apply meta-review feedback to the ideas
                finalized_ideas = []
                for idea in original_ideas:
                    # Create a copy of the idea
                    finalized_idea = copy.deepcopy(idea)
                    
                    # Add meta-review notes
                    idea_name = idea.get("Name", "")
                    specific_feedback = feedback_by_name.get(idea_name, {})
                    strengths = specific_feedback.get("Strengths", "No specific strengths identified")
                    weaknesses = specific_feedback.get("Weaknesses", "No specific weaknesses identified")
                    improvements = specific_feedback.get("SuggestedImprovements", "No specific improvements suggested")
                    
                    meta_review_notes = (
                        f"\n\nMeta-Review Notes:\n"
                        f"Strengths: {strengths}\n"
                        f"Weaknesses: {weaknesses}\n"
                        f"Suggested Improvements: {improvements}\n\n"
                        f"Overall Meta-Review Summary:\n"
                        f"Common Themes: {common_themes}\n"
                        f"Potential Synergies: {potential_synergies}\n"
                        f"Overall Assessment: {overall_assessment}\n"
                        f"Recommendations: {recommendations}"
                    )
                    
                    finalized_idea["Notes"] = finalized_idea.get("Notes", "") + meta_review_notes
                    
                    finalized_ideas.append(finalized_idea)
                
                return finalized_ideas
            else:
                # Fallback if no JSON could be extracted
                print("Warning: Could not extract JSON from Meta-Review Agent response")
                return original_ideas
                
        except Exception as e:
            print(f"Error processing Meta-Review Agent response: {e}")
            # Return the original ideas as fallback
            return original_ideas