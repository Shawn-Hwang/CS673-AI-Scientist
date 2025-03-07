import json
from itertools import combinations

def parse_json_to_ideas(file_path):
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create a list of (idea, idx) pairs where:
    # - idea is the concatenation of Title and Experiment
    # - idx is the Name field
    idea_lst = []
    for item in data:
        idea = f"{item['Title']}: {item['Experiment']}"
        name = item["Name"]
        idea_lst.append((idea, name))
    
    return idea_lst

def create_tournament_pairs(idea_list):
    """
    Takes the output of parse_json_to_ideas() and creates pairs where
    every idea is paired with every other idea exactly once.
    
    Args:
        idea_list: A list of (idea_description, idea_id) tuples
    
    Returns:
        A list of pairs, where each pair contains two (idea_description, idea_id) tuples
    """
    # Use itertools.combinations to generate all unique pairs
    # This is equivalent to a round-robin tournament pairing
    tournament_pairs = list(combinations(idea_list, 2))
    
    return tournament_pairs

if __name__ == "__main__":
    # Example usage
    file_path = "/home/huang717/CS673-AI-Scientist/templates/nanoGPT/ideas.json"
    ideas = parse_json_to_ideas(file_path)
    paired_ideas = create_tournament_pairs(ideas)

    # Print the results
    for i, pair in enumerate(paired_ideas):
        print(f"Pair {i+1}:")
        idea1, id1 = pair[0]
        idea2, id2 = pair[1]
        print(f"  Idea 1: {id1}")
        print(f"  Idea 2: {id2}")
        print()

    # To show the total number of pairs
    print(f"Total number of unique pairs: {len(paired_ideas)}")