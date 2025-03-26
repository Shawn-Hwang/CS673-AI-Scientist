from utils.llm_utils import generate_response, summarize_text
from utils.data_utils import create_idea_template

class GenerationAgent:
    """
    Generates novel research hypotheses and proposals.
    """

    def __init__(self, experiment_code, research_goal=None, skip_lit_review=False):
        self.experiment_code = experiment_code
        self.research_goal = research_goal
        self.skip_lit_review = skip_lit_review

    def generate_ideas(self, num_ideas=3):
        """Generates a list of research ideas."""
        ideas = []
        for i in range(num_ideas):
            idea = self._generate_single_idea(i + 1)
            if idea:
                ideas.append(idea)
        return ideas

    def _generate_single_idea(self, idea_num):
        """Generates a single research idea."""
        idea = create_idea_template()
        idea["Name"] = f"idea_{idea_num}"

        prompt = f"""
        You are a creative machine learning researcher.
        Your goal is to generate novel ideas for modifying the following experiment code
        to improve its performance or explore new research directions.

        Here is the experiment code:
        ```python
        {self.experiment_code}
        ```

        """

        if self.research_goal:
            prompt += f"The scientist has provided the following research goal: {self.research_goal}\n"

        if not self.skip_lit_review:
            prompt += "First, explore relevant literature using web search and summarize prior work.  Focus on recent advances related to this code.\n"
        else:
            prompt += "Skip the literature review step.\n"

        prompt += f"""
        Now, generate a novel idea for modifying the experiment.
        Give the idea a title in the style of a research paper title.
        Explain the idea in detail, including how to implement it by modifying the experiment.py code.
        Also, give the idea an interestingness score between 1 and 10, a feasibility score between 1 and 10,
        and a novelty score between 1 and 10.  Explain your reasoning for these scores in the notes.

        Format your output as a JSON object in the following format:
        {{
            "Title": "A title for the idea in the style of the title of a research paper",
            "Experiment": "Explain the idea and how to implement it",
            "Interestingness": 5,
            "Feasibility": 5,
            "Novelty": 5,
            "Notes": "Give additional thoughts about the idea and justification for why it was decided on"
        }}
        """

        response = generate_response(prompt, model="gpt-4", max_tokens=1000)
        if response:
            try:
                import json
                idea_details = json.loads(response)
                idea["Title"] = idea_details.get("Title", "")
                idea["Experiment"] = idea_details.get("Experiment", "")
                idea["Interestingness"] = idea_details.get("Interestingness", 5)
                idea["Feasibility"] = idea_details.get("Feasibility", 5)
                idea["Novelty"] = idea_details.get("Novelty", 5)
                idea["Notes"] = idea_details.get("Notes", "")
                return idea
            except json.JSONDecodeError:
                print(f"Error decoding JSON response: {response}")
                return None
        else:
            return None