import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS

NUM_REFLECTIONS = 1

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="gemini-2.0-flash",
        default="gemini-1.5-flash",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    # parser.add_argument(
    #     "--writeup",
    #     type=str,
    #     default="latex",
    #     choices=["latex"],
    #     help="What format to use for writeup",
    # )
    # parser.add_argument(
    #     "--parallel",
    #     type=int,
    #     default=0,
    #     help="Number of parallel processes to run. 0 for sequential execution.",
    # )
    # parser.add_argument(
    #     "--improvement",
    #     action="store_true",
    #     help="Improve based on reviews.",
    # )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=3,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="semanticscholar",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


if __name__ == "__main__":
    args = parse_arguments()

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    # if args.parallel > len(available_gpus):
    #     print(
    #         f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
    #     )
    #     args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )
    if not args.skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
            engine=args.engine,
        )
        novel_ideas = [idea for idea in ideas if idea["novel"]]
    
    # We don't need to save it here because the generate_ideas() function already handles that
    # with open(osp.join(base_dir, "ideas.json"), "w") as f:
    #     json.dump(ideas, f, indent=4)

    