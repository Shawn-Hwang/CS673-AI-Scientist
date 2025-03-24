import argparse
import json
import os
import os.path as osp
from pathlib import Path

from agents.generation_agent import GenerationAgent
from agents.reflection_agent import ReflectionAgent
from agents.ranking_agent import RankingAgent
from agents.evolution_agent import EvolutionAgent
from agents.meta_review_agent import MetaReviewAgent
from agents.proximity_agent import ProximityAgent
from utils import setup_genai_api

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI co-scientist for ML research idea generation")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment to analyze"
    )
    parser.add_argument(
        "--num_ideas",
        type=int,
        default=5,
        help="Number of top ideas to save"
    )
    parser.add_argument(
        "--skip_lit_review",
        action="store_true",
        help="Skip literature review step"
    )
    parser.add_argument(
        "--research_goal",
        type=str,
        default="Generate novel ideas to improve the performance or capabilities of the experiment",
        help="Research goal for the idea generation"
    )
    parser.add_argument(
        "--use_genai",
        action="store_true",
        help="Use Google Generative AI API (requires API key)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-pro",
        choices=[
            "gemini-1.5-flash", "gemini-1.5-pro", 
            "gemini-2.0-flash", "gemini-2.0-pro"
        ],
        help="Google Generative AI model to use if use_genai is True"
    )
    parser.add_argument(
        "--tournament_matches",
        type=int,
        default=10,
        help="Number of tournament matches to run"
    )
    return parser.parse_args()

def load_experiment_file(experiment_name):
    """Load the experiment.py file content"""
    experiment_path = osp.join("templates", experiment_name, "experiment.py")
    if not osp.exists(experiment_path):
        raise FileNotFoundError(f"Experiment file not found at {experiment_path}")
    
    with open(experiment_path, "r") as f:
        experiment_content = f.read()
    
    return experiment_content

def save_ideas(ideas, experiment_name, proximity_graph=None):
    """Save generated ideas to a JSON file"""
    output_path = osp.join("templates", experiment_name, "ideas.json")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(ideas, f, indent=4)
    
    print(f"Saved {len(ideas)} ideas to {output_path}")
    
    # Optionally save proximity graph for visualization
    if proximity_graph:
        graph_path = osp.join("templates", experiment_name, "proximity_graph.json")
        with open(graph_path, "w") as f:
            json.dump(proximity_graph, f, indent=4)
        print(f"Saved proximity graph to {graph_path}")

def main():
    args = parse_arguments()
    
    # Setup Gemini API if needed
    if args.use_genai:
        setup_genai_api()
    
    # Load experiment.py content
    experiment_content = load_experiment_file(args.experiment)
    print(f"Loaded experiment file for {args.experiment}")
    
    # Step 1: Generate initial ideas
    print("\n=== Step 1: Generating Initial Ideas ===")
    generation_agent = GenerationAgent(use_genai=args.use_genai, model=args.model)
    initial_ideas = generation_agent.generate_ideas(
        experiment_content=experiment_content,
        research_goal=args.research_goal,
        skip_lit_review=args.skip_lit_review
    )
    print(f"Generated {len(initial_ideas)} initial ideas")
    
    # Step 2: Review ideas
    print("\n=== Step 2: Reviewing Ideas ===")
    reflection_agent = ReflectionAgent(use_genai=args.use_genai, model=args.model)
    reviewed_ideas = reflection_agent.review_ideas(
        ideas=initial_ideas,
        experiment_content=experiment_content,
        research_goal=args.research_goal
    )
    print(f"{len(reviewed_ideas)} ideas passed review")
    
    # Step 3: Calculate proximity between ideas
    print("\n=== Step 3: Calculating Idea Proximity ===")
    proximity_agent = ProximityAgent(use_genai=args.use_genai, model=args.model)
    ideas_with_proximity, proximity_graph = proximity_agent.calculate_proximity(
        ideas=reviewed_ideas,
        experiment_content=experiment_content,
        research_goal=args.research_goal
    )
    print(f"Calculated proximity for {len(ideas_with_proximity)} ideas")
    
    # Step 4: Rank ideas
    print("\n=== Step 4: Ranking Ideas ===")
    ranking_agent = RankingAgent(use_genai=args.use_genai, model=args.model)
    ranked_ideas = ranking_agent.rank_ideas(
        ideas=ideas_with_proximity,
        experiment_content=experiment_content,
        research_goal=args.research_goal,
        num_matches=args.tournament_matches
    )
    print(f"Ranked {len(ranked_ideas)} ideas")
    
    # Step 5: Evolve top ideas
    print("\n=== Step 5: Evolving Top Ideas ===")
    evolution_agent = EvolutionAgent(use_genai=args.use_genai, model=args.model)
    evolved_ideas = evolution_agent.evolve_ideas(
        ideas=ranked_ideas,
        experiment_content=experiment_content,
        research_goal=args.research_goal
    )
    print(f"Evolved {len(evolved_ideas)} ideas")
    
    # Step 6: Recalculate proximity including evolved ideas
    print("\n=== Step 6: Recalculating Proximity with Evolved Ideas ===")
    all_ideas = ranked_ideas + evolved_ideas
    all_ideas_with_proximity, updated_proximity_graph = proximity_agent.calculate_proximity(
        ideas=all_ideas,
        experiment_content=experiment_content,
        research_goal=args.research_goal
    )
    
    # Step 7: Final ranking
    print("\n=== Step 7: Final Ranking ===")
    final_ranked_ideas = ranking_agent.rank_ideas(
        ideas=all_ideas_with_proximity,
        experiment_content=experiment_content,
        research_goal=args.research_goal,
        num_matches=args.tournament_matches
    )
    
    # Step 8: Meta-review for final feedback
    print("\n=== Step 8: Meta-Review ===")
    meta_review_agent = MetaReviewAgent(use_genai=args.use_genai, model=args.model)
    finalized_ideas = meta_review_agent.finalize_ideas(
        ideas=final_ranked_ideas[:min(args.num_ideas, len(final_ranked_ideas))],
        experiment_content=experiment_content,
        research_goal=args.research_goal
    )
    
    # Save the top ideas
    save_ideas(finalized_ideas, args.experiment, updated_proximity_graph)
    
    # Print top ideas summary
    print("\n=== Top Ideas Generated ===")
    for i, idea in enumerate(finalized_ideas):
        print(f"{i+1}. {idea['Title']} (ELO: {idea['ELO rating']})")
    
    print("\nAI Co-scientist completed successfully!")

if __name__ == "__main__":
    main()

# Example usage: python main.py --experiment experiment_name --num_ideas 5 --use_genai --model gemini-2.0-flash