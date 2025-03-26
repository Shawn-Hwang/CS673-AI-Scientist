from utils.llm_utils import generate_response

class ReflectionAgent:
    """
    Critically examines the correctness, quality, and novelty of generated hypotheses.
    """

    def __init__(self, experiment_code):
        self.experiment_code = experiment_code

    def review_idea(self, idea):
        """Reviews a single idea."""
        print(f"Reviewing idea: {idea['Name']}")

        # Initial Review
        initial_review = self._initial_review(idea)
        print(f"Initial Review: {initial_review}")
        if "reject" in initial_review.lower():
            idea["Notes"] += f"\nRejected after initial review: {initial_review}"
            return idea

        # Full Review
        full_review = self._full_review(idea)
        idea["Notes"] += f"\nFull Review: {full_review}"

        # Deep Verification Review
        deep_verification_review = self._deep_verification_review(idea)
        idea["Notes"] += f"\nDeep Verification Review: {deep_verification_review}"

        # Observation Review (simplified)
        observation_review = self._observation_review(idea)
        idea["Notes"] += f"\nObservation Review: {observation_review}"

        # Simulation Review (simplified)
        simulation_review = self._simulation_review(idea)
        idea["Notes"] += f"\nSimulation Review: {simulation_review}"

        return idea

    def _initial_review(self, idea):
        """Performs an initial review assessing correctness, quality, and novelty."""
        prompt = f"""
        You are a scientific peer reviewer.  Critically examine the following research idea for correctness, quality, and novelty.
        If the idea has obvious flaws, is not novel, or is clearly unsuitable, reject it.

        Here is the idea:
        Title: {idea['Title']}
        Experiment: {idea['Experiment']}

        Based on your initial review, is this idea worth pursuing?  Explain your reasoning.
        Answer 'Reject' if the idea should be rejected.
        """
        return generate_response(prompt, max_tokens=300)

    def _full_review(self, idea):
        """Performs a full review, leveraging external tools and web searches."""
        prompt = f"""
        You are a scientific peer reviewer conducting a full review.
        Critically examine the following research idea for correctness, quality, and novelty, using literature search.

        Here is the experiment code:
        ```python
        {self.experiment_code}
        ```

        Here is the idea:
        Title: {idea['Title']}
        Experiment: {idea['Experiment']}

        Provide a detailed review, summarizing known aspects of the hypothesis and judging its novelty based on existing literature.
        """
        return generate_response(prompt, max_tokens=500)

    def _deep_verification_review(self, idea):
        """Conducts a deep verification review, decomposing the hypothesis into constituent assumptions."""
        prompt = f"""
        You are a scientific peer reviewer conducting a deep verification review.
        Decompose the following research idea into its constituent assumptions.
        Break down each assumption into fundamental sub-assumptions and independently evaluate them for correctness.

        Here is the experiment code:
        ```python
        {self.experiment_code}
        ```

        Here is the idea:
        Title: {idea['Title']}
        Experiment: {idea['Experiment']}

        Summarize any reasons for potential hypothesis invalidation due to incorrect assumptions.
        """
        return generate_response(prompt, max_tokens=500)

    def _observation_review(self, idea):
        """Explores whether the hypothesis can account for long-tail observations from prior experimental results."""
        prompt = f"""
        You are a scientific peer reviewer.
        Explore whether the following research idea can account for long-tail observations from prior experimental results.

        Here is the experiment code:
        ```python
        {self.experiment_code}
        ```

        Here is the idea:
        Title: {idea['Title']}
        Experiment: {idea['Experiment']}

        Assess if the hypothesis is a superior explanation over existing ones, assuming its validity.
        """
        return generate_response(prompt, max_tokens=300)

    def _simulation_review(self, idea):
        """Reviews the hypothesis by simulating it in a step-wise fashion."""
        prompt = f"""
        You are a scientific peer reviewer.
        Review the following research idea by simulating it in a step-wise fashion.
        Identify and summarize potential failure scenarios.

        Here is the experiment code:
        ```python
        {self.experiment_code}
        ```

        Here is the idea:
        Title: {idea['Title']}
        Experiment: {idea['Experiment']}
        """
        return generate_response(prompt, max_tokens=300)