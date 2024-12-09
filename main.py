import argparse
import os
import json
from settings.config import DEFAULT_MODEL_NAME
from manager.metrics_manager import MetricsManager
from evaluators.review import SentimentReviewEvaluator
from evaluators.translation import TranslationQualityEvaluator

# Map task names to evaluator classes
task_map = {
    "review": SentimentReviewEvaluator,
    "translation": TranslationQualityEvaluator,
}

def parse_args():
    """
    Parses command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: Parsed arguments containing model, data_path, save_dir, and enable_bleu_rouge.
    """
    parser = argparse.ArgumentParser(description="Evaluate tasks using LLM-based evaluators.")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name or path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--data-path",
        "-data",
        type=str,
        required=True,
        help="Path to the JSON data file containing task inputs."
        "Ensure that the JSON file name follows the format ‘{task_name}.json’.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results).",
    )
    parser.add_argument(
        "--enable-bleu-rouge",
        action="store_true",
        help="Enable BLEU and ROUGE score calculation (applicable for translation tasks).",
    )

    return parser.parse_args()


def main():
    """
    Main entry point for running evaluations.
    """
    # Parse command-line arguments
    args = parse_args()

    # Validate task type
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}")

    # Load data
    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Infer task type from file name
    task = args.data_path.split("/")[-1].split(".")[0]

    if task not in task_map:
        raise ValueError(f"Unsupported task type inferred from file: {task}")

    if args.enable_bleu_rouge:
        assert task == "translation", "This argument is applicable only to translation tasks."

    # Initialize MetricsManager with BLEU and ROUGE flag
    metrics_manager = MetricsManager(enable_bleu_rouge=args.enable_bleu_rouge)

    # Instantiate the appropriate evaluator
    evaluator_class = task_map[task]
    evaluator = evaluator_class(model_name=args.model, metrics_manager=metrics_manager)

    # Perform evaluation
    print(f"Evaluating task: {task}")
    results = evaluator.evaluate(data=data)

    # Prepare save directory
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{task}_evaluation_results.json")

    # Save results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation complete. Results saved to {save_path}")


if __name__ == "__main__":
    main()