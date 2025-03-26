from utils.llm_utils import generate_response

class RankingAgent:
    """
    Uses an Elo-based tournament to automatically evaluate and rank hypotheses.
    """

    def __init__(self, experiment_code):
        self.experiment_code = experiment_code

    def conduct_debate(self, idea1, idea2):
        """Conducts a simulated scientific debate between two ideas."""
        prompt = f"""
        You are a moderator in a scientific debate between two research ideas.
        Your goal is to determine which idea is better based on novelty, correctness, and testability.

        Here is the experiment code:
        ```python
        {self.experiment_code}
        ```

        Here is idea 1:
        Title: {idea1['Title']}
        Experiment: {idea1['Experiment']}

        Here is idea 2:
        Title: {idea2['Title']}
        Experiment: {idea2['Experiment']}

        Engage in a multi-turn scientific debate, probing the strengths and weaknesses of each idea.
        Conclude with a decision regarding which hypothesis is better.
        Answer with "idea1", "idea2", or "draw".
        """
        return generate_response(prompt, max_tokens=500)