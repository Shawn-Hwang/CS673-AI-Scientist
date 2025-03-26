import json
import os

def load_experiment_code(experiment_name):
    """Loads the code from the experiment.py file."""
    filepath = os.path.join("templates", experiment_name, "experiment.py")
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: experiment.py not found for experiment '{experiment_name}'")
        return None
    except Exception as e:
        print(f"Error loading experiment code: {e}")
        return None

def load_ideas(experiment_name):
    """Loads existing ideas from ideas.json, if it exists."""
    filepath = os.path.join("templates", experiment_name, "ideas.json")
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []  # No existing ideas
    except json.JSONDecodeError:
        print("Error: Invalid JSON in ideas.json.  Starting with an empty list.")
        return []
    except Exception as e:
        print(f"Error loading ideas: {e}")
        return []

def save_ideas(experiment_name, ideas):
    """Saves the ideas to a JSON file."""
    filepath = os.path.join("templates", experiment_name, "ideas.json")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        with open(filepath, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"Ideas saved to {filepath}")
    except Exception as e:
        print(f"Error saving ideas: {e}")

def create_idea_template():
    """Returns a template for a new idea."""
    return {
        "Name": "",
        "Title": "",
        "Experiment": "",
        "Interestingness": 5,
        "Feasibility": 5,
        "Novelty": 5,
        "ELO rating": 1200,
        "Notes": ""
    }