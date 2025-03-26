import networkx as nx
from utils.llm_utils import generate_response
import numpy as np

class ProximityAgent:
    """
    Calculates the similarity between research hypotheses and proposals and builds a proximity graph.
    """

    def __init__(self, experiment_code):
        # self.graph = nx.Graph()
        self.experiment_code = experiment_code

    def build_graph(self, ideas):
        """Builds a proximity graph based on the similarity between ideas."""
        self.graph = np.zeros((len(ideas), len(ideas)))
        for i in range(len(ideas)):
            for j in range(i + 1, len(ideas)):
                idea1 = ideas[i]
                idea2 = ideas[j]
                similarity = self._calculate_similarity(idea1, idea2)
                # if similarity > 0.7:  # Threshold for connecting ideas in the graph
                #     self.graph.add_edge(idea1["Name"], idea2["Name"], weight=similarity)
                self.graph[i, j] = similarity
                self.graph[j, i] = similarity

    def _calculate_similarity(self, idea1, idea2):
        """Calculates the similarity between two ideas using a language model."""
        prompt = f"""
        You are a research assistant.
        Calculate the similarity between the following two research ideas on a scale of 0 to 1.
        The ideas will be modifying the following template:
        {self.experiment_code}

        Idea 1:
        Title: {idea1['Title']}
        Experiment: {idea1['Experiment']}

        Idea 2:
        Title: {idea2['Title']}
        Experiment: {idea2['Experiment']}

        Provide a similarity score between 0 and 1.  Explain your reasoning.
        End your response with a colon (:) followed by the similarity score.
        """
        response = generate_response(prompt, max_tokens=200)
        try:
            similarity = float(response.split(":")[-1].strip())  # Extract the similarity score
            return similarity
        except:
            return 0.5