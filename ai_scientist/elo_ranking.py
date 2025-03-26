
import json
import csv
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional
import time
from parse_ideas import parse_json_to_ideas, create_tournament_pairs
from pairwise_comparison import compare_two_ideas
from llm import create_client
import os


def record_comparison_results(winner_id: str, loser_id: str, elo_ratings: Dict[str, float], 
                              comparison_history: List[Dict], k_factor: int = 32,
                              round_number: int = 0) -> Dict[str, float]:
    """
    Record the result of a comparison and update ELO ratings
    
    Args:
        winner_id: The ID of the winning idea
        loser_id: The ID of the losing idea
        elo_ratings: Current ELO ratings dictionary
        comparison_history: List to store comparison results
        k_factor: The K-factor for ELO calculation (how much ratings change)
        
    Returns:
        Updated ELO ratings dictionary
    """
    # Get current ratings
    winner_rating = elo_ratings[winner_id]
    loser_rating = elo_ratings[loser_id]
    
    # Calculate expected scores
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
    
    # Calculate new ratings
    new_winner_rating = winner_rating + k_factor * (1 - expected_winner)
    new_loser_rating = loser_rating + k_factor * (0 - expected_loser)
    
    # Update the ratings
    elo_ratings[winner_id] = new_winner_rating
    elo_ratings[loser_id] = new_loser_rating
    
    # Record the comparison result
    comparison_history.append({
        'round': round_number,
        'winner': winner_id,
        'loser': loser_id,
        'winner_old_rating': winner_rating,
        'loser_old_rating': loser_rating,
        'winner_new_rating': new_winner_rating,
        'loser_new_rating': new_loser_rating
    })
    
    return elo_ratings

def calculate_elo_rankings(idea_list: List[Tuple[str, str]], client, model, experiment,
                          initial_rating: int = 1400, k_factor: int = 32) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Calculate ELO rankings for a list of ideas
    
    Args:
        idea_list: List of (idea_description, idea_id) tuples
        client: LLM API client
        model: LLM model to use
        initial_rating: Starting ELO rating for each idea
        k_factor: K-factor for ELO calculations
        
    Returns:
        Tuple of (final ELO ratings dictionary, comparison history list)
    """
    # Initialize ELO ratings
    elo_ratings = {idea_id: initial_rating for _, idea_id in idea_list}
    
    # Create tournament pairs
    tournament_pairs = create_tournament_pairs(idea_list)
    
    # Store comparison history
    comparison_history = []
    
    # Compare each pair and update ratings
    total_comparisons = len(tournament_pairs)
    for i, pair in enumerate(tournament_pairs):
        idea_1, idea_2 = pair
        
        # Start timing the comparison
        start_time = time.time()
        
        winner_id, loser_id = compare_two_ideas(idea_1, idea_2, client, model, experiment)
        
        # Calculate the time taken
        comparison_time = time.time() - start_time
        
        if winner_id and loser_id:
            elo_ratings = record_comparison_results(
                winner_id, loser_id, elo_ratings, comparison_history, k_factor, 
                round_number=i+1  # Pass the round number (1-based)
            )
            print(f"Comparison done: {i+1}/{total_comparisons} (Time: {comparison_time:.2f} seconds) - Winner: {winner_id}")
        else:
            print(f"Comparison {i+1}/{total_comparisons} (Time: {comparison_time:.2f} seconds) - FAILED to determine winner/loser")
    
    return elo_ratings, comparison_history


def save_results_to_csv(elo_ratings: Dict[str, float], comparison_history: List[Dict], 
                       experiment: str):
    """
    Save ELO ratings and comparison history to CSV files
    
    Args:
        elo_ratings: Dictionary of final ELO ratings
        comparison_history: List of comparison results
        ratings_file: Filename for ratings CSV
        history_file: Filename for history CSV
    """
    # Save final rankings
    with open(os.path.join("templates", experiment, "idea_rankings.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idea_id', 'elo_rating'])
        
        # Sort by rating in descending order
        sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        for idea_id, rating in sorted_ratings:
            writer.writerow([idea_id, rating])
    
    # Save comparison history
    if comparison_history:
        try:
            with open(os.path.join("templates", experiment, "comparison_history.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=comparison_history[0].keys())
                writer.writeheader()
                writer.writerows(comparison_history)
            comparison_file_path = os.path.join("templates", experiment, "comparison_history.csv")
            print(f"Comparison history saved to {comparison_file_path}")
        except Exception as e:
            print(f"Error saving comparison history: {e}")
    else:
        print("No comparison history to save - history list is empty")
    
    ranking_file_path = os.path.join("templates", experiment, "idea_rankings.csv")
    print(f"Final rankings saved to {ranking_file_path}")

# Main function to run the entire process
def run_elo_tournament(experiment: str, client, model, max_ideas=3):
    """
    Run the complete ELO ranking process
    
    Args:
        json_file_path: Path to the JSON file with ideas
        client: LLM API client
        model: LLM model to use
    """
    # Step 1: Parse JSON to ideas
    idea_list = parse_json_to_ideas(os.path.join("templates", experiment, "ideas.json"))
    if max_ideas:
        idea_list = idea_list[:max_ideas]
    
    # Steps 4-5: Calculate ELO rankings
    elo_ratings, comparison_history = calculate_elo_rankings(idea_list, client, model, experiment)
    
    # Save results
    save_results_to_csv(elo_ratings, comparison_history, experiment)
    
    return elo_ratings, comparison_history

if __name__ == "__main__":
    # parse the --experiment argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="nanoGPT")
    args = parser.parse_args()
    experiment = args.experiment

    # Create client
    client, client_model = create_client("gemini-2.0-flash")

    run_elo_tournament(experiment, client, client_model, max_ideas=None)




    