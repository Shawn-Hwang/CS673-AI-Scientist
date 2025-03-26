import trueskill
import random

class Tournament:
    """
    Manages the Elo-based tournament for ranking hypotheses.
    Uses the TrueSkill algorithm for more robust ranking.
    """

    def __init__(self, initial_rating=1200, mu=25.0, sigma=8.333, beta=250.0, tau=8.333/100, draw_probability=0.0):
        self.env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_probability)
        self.ratings = {}  # Store ratings (TrueSkill objects)
        self.initial_rating_value = initial_rating

    def add_idea(self, idea_name):
        """Adds a new idea to the tournament with an initial rating."""
        if idea_name not in self.ratings:
            self.ratings[idea_name] = self.env.Rating(mu=self.env.mu, sigma=self.env.sigma)

    def get_rating(self, idea_name):
        """Returns the rating (TrueSkill object) for an idea."""
        return self.ratings.get(idea_name)

    def get_elo_rating(self, idea_name):
        """Returns the approximate ELO rating for an idea."""
        rating = self.get_rating(idea_name)
        if rating:
            return self.env.expose(rating)
        else:
            return None

    def conduct_match(self, idea1_name, idea2_name, winner=None):
        """
        Conducts a match between two ideas and updates their ratings.

        Args:
            idea1_name (str): Name of the first idea.
            idea2_name (str): Name of the second idea.
            winner (str):  'idea1', 'idea2', or 'draw'.  None if the match hasn't happened.
        """
        if idea1_name not in self.ratings or idea2_name not in self.ratings:
            raise ValueError("One or both ideas not found in the tournament.")

        rating1 = self.ratings[idea1_name]
        rating2 = self.ratings[idea2_name]

        if winner == 'idea1':
            new_rating1, new_rating2 = self.env.rate_1vs1(rating1, rating2)
        elif winner == 'idea2':
            new_rating2, new_rating1 = self.env.rate_1vs1(rating2, rating1)
        elif winner == 'draw':
            new_rating1, new_rating2 = self.env.rate_1vs1(rating1, rating2, drawn=True)
        else:
            return  # No update if there's no winner

        self.ratings[idea1_name] = new_rating1
        self.ratings[idea2_name] = new_rating2

    def get_ranked_ideas(self):
        """Returns a list of ideas ranked by their Elo rating (descending)."""
        ranked_ideas = sorted(self.ratings.items(), key=lambda item: self.env.expose(item[1]), reverse=True)
        return ranked_ideas

    def choose_match(self, ideas, proximity_graph=None):
        """
        Chooses a match between two ideas based on proximity and ranking.

        Args:
            ideas (list): A list of idea names to choose from.
            proximity_graph (networkx.Graph): A graph representing the similarity between ideas.

        Returns:
            tuple: A tuple containing the names of the two ideas to match, or None if no match can be found.
        """
        if len(ideas) < 2:
            return None

        # Prioritize matches between similar ideas based on the proximity graph
        if proximity_graph:
            for _ in range(10):  # Try a few times to find a good match
                idea1 = random.choice(ideas)
                neighbors = list(proximity_graph.neighbors(idea1))
                if neighbors:
                    idea2 = random.choice(neighbors)
                    if idea2 in ideas:
                        return idea1, idea2

        # If no similar ideas are found, choose two random ideas
        idea1 = random.choice(ideas)
        ideas_without_idea1 = [idea for idea in ideas if idea != idea1]
        if ideas_without_idea1:
            idea2 = random.choice(ideas_without_idea1)
            return idea1, idea2
        else:
            return None