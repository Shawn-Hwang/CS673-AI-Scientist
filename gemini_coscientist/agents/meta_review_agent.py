from utils.llm_utils import generate_response

class MetaReviewAgent:
    """
    Synthesizes insights from all reviews and tournament debates to optimize other agents' performance.
    """

    def __init__(self):
        pass

    def generate_meta_review(self, ideas):
        """Generates a meta-review based on the reviews of the ideas."""
        reviews = "\n".join([idea["Notes"] for idea in ideas])
        prompt = f"""
        You are a meta-reviewer. Synthesize insights from the following reviews to identify recurring patterns and common issues.

        Reviews:
        {reviews}

        Provide a meta-review critique summarizing the common patterns and issues.
        """
        return generate_response(prompt, max_tokens=500)

    def generate_research_overview(self, ideas):
        """Generates a research overview based on the top-ranked hypotheses."""
        top_ideas = sorted(ideas, key=lambda x: x["ELO rating"], reverse=True)[:3]  # Get top 3 ideas
        top_ideas_summary = "\n".join([f"{idea['Title']}: {idea['Experiment']}" for idea in top_ideas])

        prompt = f"""
        You are a research strategist. Synthesize the following top-ranked research ideas into a research overview,
        providing a roadmap for future research.

        Top Ideas:
        {top_ideas_summary}

        Outline potential research areas and directions relevant to the research goal, justifying their importance
        and suggesting specific experiments within each.
        """
        return generate_response(prompt, max_tokens=500)