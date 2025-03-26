from agents.generation_agent import GenerationAgent
from agents.reflection_agent import ReflectionAgent
from agents.ranking_agent import RankingAgent
from agents.proximity_agent import ProximityAgent
from agents.evolution_agent import EvolutionAgent
from agents.meta_review_agent import MetaReviewAgent
from core.context_memory import ContextMemory
from core.tournament import Tournament
from utils.data_utils import load_ideas, save_ideas

import networkx as nx
import random
from tqdm import tqdm  # For progress bars

class Supervisor:
    """
    Orchestrates the AI co-scientist system.
    """

    def __init__(self, experiment_name, research_goal=None, num_ideas=5, skip_lit_review=False):
        self.experiment_name = experiment_name
        self.research_goal = research_goal
        self.num_ideas = num_ideas
        self.skip_lit_review = skip_lit_review
        self.context_memory = ContextMemory()
        self.experiment_code = self.load_experiment_code()
        self.generation_agent = GenerationAgent(self.experiment_code, self.research_goal, self.skip_lit_review)
        self.reflection_agent = ReflectionAgent(self.experiment_code)
        self.ranking_agent = RankingAgent(self.experiment_code)
        self.proximity_agent = ProximityAgent(self.experiment_code)
        self.evolution_agent = EvolutionAgent(self.experiment_code)
        self.meta_review_agent = MetaReviewAgent()
        self.tournament = Tournament()
        self.ideas = load_ideas(experiment_name)
        self.proximity_graph = nx.Graph()

        # Load existing ideas into the tournament
        for idea in self.ideas:
            self.tournament.add_idea(idea["Name"])
            self.tournament.ratings[idea["Name"]] = self.tournament.env.Rating(mu=idea["ELO rating"] / 36.667 + self.tournament.env.mu, sigma=self.tournament.env.sigma)

    def load_experiment_code(self):
        """Loads the experiment code from the experiment.py file."""
        from utils.data_utils import load_experiment_code
        return load_experiment_code(self.experiment_name)

    def run(self, iterations=3):
        """Runs the AI co-scientist system for a specified number of iterations."""
        for i in range(iterations):
            print(f"Iteration {i + 1}/{iterations}")

            # 1. Generate New Ideas
            new_ideas = self.generation_agent.generate_ideas(num_ideas=self.num_ideas)
            self.ideas.extend(new_ideas)
            for idea in new_ideas:
                self.tournament.add_idea(idea["Name"])

            # 2. Review Ideas
            for idea in tqdm(self.ideas, desc="Reviewing Ideas"):
                if "Notes" not in idea:
                    idea["Notes"] = ""
                idea = self.reflection_agent.review_idea(idea)

            # 3. Build Proximity Graph
            self.proximity_agent.build_graph(self.ideas)
            self.proximity_graph = self.proximity_agent.graph

            # 4. Conduct Tournament Matches
            num_matches = min(10, len(self.ideas) * (len(self.ideas) - 1) // 2) # Number of matches to run
            for _ in tqdm(range(num_matches), desc="Conducting Tournament Matches"):
                match = self.tournament.choose_match([idea["Name"] for idea in self.ideas], self.proximity_graph)
                if match:
                    idea1_name, idea2_name = match
                    idea1 = next(idea for idea in self.ideas if idea["Name"] == idea1_name)
                    idea2 = next(idea for idea in self.ideas if idea["Name"] == idea2_name)
                    winner = self.ranking_agent.conduct_debate(idea1, idea2)

                    self.tournament.conduct_match(idea1_name, idea2_name, winner)

            # Update ELO ratings in ideas list
            for idea in self.ideas:
                idea["ELO rating"] = self.tournament.get_elo_rating(idea["Name"])

            # 5. Evolve Top Ideas
            ranked_ideas = sorted(self.ideas, key=lambda x: x["ELO rating"], reverse=True)
            for idea in ranked_ideas[:2]:  # Evolve top 2 ideas
                evolved_idea = self.evolution_agent.evolve_idea(idea)
                if evolved_idea:
                    self.ideas.append(evolved_idea)
                    self.tournament.add_idea(evolved_idea["Name"])

            # 6. Meta-Review
            meta_review = self.meta_review_agent.generate_meta_review(self.ideas)
            print(f"Meta-Review: {meta_review}")

            research_overview = self.meta_review_agent.generate_research_overview(self.ideas)
            print(f"Research Overview: {research_overview}")

            # 7. Save Ideas
            save_ideas(self.experiment_name, self.ideas)

        print("AI Co-scientist run complete.")

    def get_top_ideas(self, num_ideas):
        """Returns the top-ranked ideas."""
        ranked_ideas = sorted(self.ideas, key=lambda x: x["ELO rating"], reverse=True)
        return ranked_ideas[:num_ideas]
    