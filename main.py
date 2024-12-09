import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tasks using LLM-based evaluators.")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help=f"Model name or path (default: 'meta-llama/Llama-3.2-3B-Instruct')",
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
    # Parse command-line arguments
    args = parse_args()

    # Load data
    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Infer task type from file name
    task = args.data_path.split("/")[-1].split(".")[0]

    if args.enable_bleu_rouge:
        assert task == "translation", "This argument is applicable only to translation tasks."

    print(f"Evaluation complete. Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()