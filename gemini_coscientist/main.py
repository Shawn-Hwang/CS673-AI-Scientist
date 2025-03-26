import argparse
from core.supervisor import Supervisor
from utils.data_utils import load_experiment_code

def main():
    parser = argparse.ArgumentParser(description="AI Co-scientist for Machine Learning Research")
    parser.add_argument("--experiment", required=True, help="The name of the experiment (directory in templates/)")
    parser.add_argument("--num_ideas", type=int, default=5, help="Number of ideas to generate per iteration")
    parser.add_argument("--skip_lit_review", action="store_true", help="Skip literature review steps")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations to run the co-scientist")
    parser.add_argument("--research_goal", type=str, default=None, help="Optional research goal provided by the scientist")

    args = parser.parse_args()

    experiment_code = load_experiment_code(args.experiment)
    if not experiment_code:
        return

    supervisor = Supervisor(
        experiment_name=args.experiment,
        research_goal=args.research_goal,
        num_ideas=args.num_ideas,
        skip_lit_review=args.skip_lit_review,
    )
    supervisor.run(iterations=args.iterations)

    top_ideas = supervisor.get_top_ideas(args.num_ideas)
    print("\nTop Ideas:")
    for idea in top_ideas:
        print(f"  - {idea['Title']} (ELO: {idea['ELO rating']})")

if __name__ == "__main__":
    main()