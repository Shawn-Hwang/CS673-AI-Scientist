from utils.llm_utils import generate_response
from utils.data_utils import create_idea_template

class EvolutionAgent:
    """
    Continuously refines and improves existing hypotheses and proposals.
    """

    def __init__(self, experiment_code):
        self.experiment_code = experiment_code

    def evolve_idea(self, idea):
        """Evolves a single idea."""
        print(f"Evolving idea: {idea['Name']}")
        evolved_idea = create_idea_template()
        evolved_idea["Name"] = f"{idea['Name']}_evolved"

        prompt = f"""
        You are an experienced machine learning researcher.
        Your goal is to refine and improve the following research idea.

        Here is the experiment code:
        ```python
        {self.experiment_code}
        ```

        Here is the idea:
        Title: {idea['Title']}
        Experiment: {idea['Experiment']}

        Suggest improvements to the idea, elaborate on details to fill reasoning gaps,
        and address any potential weaknesses.
        Generate a new idea based on the improved idea.

        Format your output as a JSON object in the following format:
        {{
            "Title": "A title for the improved idea in the style of a research paper title",
            "Experiment": "Explain the improved idea and how to implement it",
            "Interestingness": 5,
            "Feasibility": 5,
            "Novelty": 5,
            "Notes": "Give additional thoughts about the idea and justification for why it was decided on"
        }}
        """

        response = generate_response(prompt, max_tokens=1000)
        if response:
            try:
                import json
                idea_details = json.loads(response)
                evolved_idea["Title"] = idea_details.get("Title", "")
                evolved_idea["Experiment"] = idea_details.get("Experiment", "")
                evolved_idea["Interestingness"] = idea_details.get("Interestingness", 5)
                evolved_idea["Feasibility"] = idea_details.get("Feasibility", 5)
                evolved_idea["Novelty"] = idea_details.get("Novelty", 5)
                evolved_idea["Notes"] = idea_details.get("Notes", "")
                return evolved_idea
            except json.JSONDecodeError:
                print(f"Error decoding JSON response: {response}")
                return None
        else:
            return None