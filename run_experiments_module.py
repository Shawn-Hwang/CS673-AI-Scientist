import json
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
import argparse
import multiprocessing
import openai
import os
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime
from ai_scientist.llm import create_client, AVAILABLE_LLMS

MAX_ITERS = 4
MAX_STDERR_OUTPUT = 1500

coder_prompt = """Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

Include any additional learnable parameters inside the Agent module. Make all the necessary changes in one iteration. 
After you complete the changes, we will run the command `python experiment.py --out_dir={idea_name}' and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS."""

def run_experiment(folder_name, idea_name, timeout=7200):
    cwd = osp.abspath(folder_name)
    # COPY CODE SO WE CAN SEE IT.
    shutil.copy(
        osp.join(folder_name, "experiment.py"),
        osp.join(folder_name, f"{idea_name}.py"),
    )

    # LAUNCH COMMAND
    command = [
        "python",
        "experiment.py",
        f"--out_dir={idea_name}",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run {idea_name} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"{idea_name}")):
                shutil.rmtree(osp.join(cwd, f"{idea_name}"))
            print(f"Run failed with the following error {result.stderr}")
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            next_prompt = ""
        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Run {idea_name} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"{idea_name}")):
            shutil.rmtree(osp.join(cwd, f"{idea_name}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt
    
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    current_iter = 0
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        idea_name=idea["Name"],
        baseline_results=baseline_results,
    )
    while True:
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break
        coder_out = coder.run(next_prompt)
        print(coder_out)
        return_code, next_prompt = run_experiment(folder_name, idea["Name"])
        if return_code == 0:
            break
        current_iter += 1
    if current_iter >= MAX_ITERS:
        print("Experiment not completed.")
        return False

    return True

def worker(
        queue,
        base_dir,
        results_dir,
        model,
        client,
        client_model,
        writeup,
        improvement,
):
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            model,
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")


def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        log_file=False,
):
    ## CREATE PROJECT FOLDER
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = idea['Name']
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    # baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    # notes = osp.join(folder_name, "notes.txt")
    # with open(notes, "w") as f:
    #     f.write(f"# Title: {idea['Title']}\n")
    #     f.write(f"# Experiment description: {idea['Experiment']}\n")
    #     f.write(f"## Run 0: Baseline\n")
    #     f.write(f"Results: {baseline_results}\n")
    #     f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
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
        default="gemini/gemini-2.0-flash",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--idea",
        type=str,
        default=None,
        help="Name of the idea to execute.",
    )
    parser.add_argument(
        "--idea_file",
        type=str,
        default="ideas.json",
        help="Path to the JSON file containing the idea.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    with open(osp.join(base_dir, args.idea_file), "r") as f:
        ideas = json.load(f)

    idea = [idea for idea in ideas if idea["Name"]  == args.idea][0]

    print(f"Processing idea: {idea['Name']}")
    try:
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            args.model,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    except Exception as e:
        print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")
        import traceback
        print(traceback.format_exc())
    print("All ideas evaluated.")
