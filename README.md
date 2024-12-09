# LLM Based Evaluator

This repository contains Pytorch implementation for evaluating various types of models using LLM as an evaluator.

## Prerequisites
Our implementation is based on Python 3.9.20.
Please see the full list of packages required to run our codes in `requirements.txt`.

- Python 3.9.20
- PyTorch 2.2.2

If you want to use an LLM from Hugging Face, you need to authenticate your account using the following command:
```bash
huggingface-cli login
```
After running this command, you will be prompted to copy and paste your access token.

You can obtain your access token from the Hugging Face website by navigating to the Access Tokens section. This can be found by clicking on the profile icon at the top right corner of the website.

## Code Structure
The project is organized as below:
```bash
project/
├── evaluators/
│   ├── base.py                  # Contains LLMBasedEvaluator
│   ├── review.py                # Contains SentimentReviewEvaluator
│   ├── translation.py           # Contains TranslationQualityEvaluator
├── managers/
│   ├── prompt_manager.py        # Contains PromptManager
│   ├── metrics_manager.py       # Contains MetricsManager
├── utils/
│   ├── metrics.py               # Contains scoring functions (e.g., accuracy, f1-score)
│   ├── helpers.py               # Contains utility functions (e.g., find_pattern)
├── data/
│   ├── system_prompts/
│   │   ├── review_prompt.txt    # Organize prompts into a subfolder.
│   │   ├── translation_prompt.txt
│   │   ├── translation_eval_prompt.txt
│   ├── review.json              # JSON data for input tasks
│   ├── translation.json
├── settings/
│   ├── config.py                # Centralized configuration for file paths and task types.
└── main.py                      # Entry point for running evaluations
```
The evaluation results for each task (sentiment review, translation quality evaluation) using the `meta-llama/Llama-3.2-1B-Instruct` model have been included in the `./results` directory.

## Usage

### Evaluate the Sentiment Review Classifier
To evaluate the sentiment review classifier using the default LLM, run one of the following commands:
```bash
python main.py --data-path ./data/review.json --save-dir ./results
```
or
```bash
python main.py -data ./data/review.json --save-dir ./results
```
**Notes:**
>The `--data-path` (or `-data`) argument is required. The JSON file specified in the argument must follow the naming convention `{task_name}.json`.

If you want to use a different LLM as the evaluator, specify the model with the following commands:
```bash
python main.py --model model_name_or_path --data-path ./data/review.json --save-dir ./results
```
or
```bash
python main.py -m model_name_or_path
```
Note that the specified model should be available on Hugging Face or downloaded locally from Hugging Face.

You can designate a directory to save the evaluation results as a JSON file:
```bash
python main.py --data-path ./data/review.json --save-dir dir_path_to_save_results
```

### Evaluate the Translation Quality Classifier
To evaluate the translation quality classifier using the default LLM, run one of the following commands:
```bash
python main.py --data-path './data/translation.json' --save-dir ./results
```
or
```bash
python main.py -data './data/translation.json' --save-dir ./results
```

If you want to use a different LLM as the evaluator, specify the model with the following commands:
```bash
python main.py --model model_name_or_path --data-path './data/translation.json' --save-dir ./results
```
or
```bash
python main.py -m model_name_or_path -data './data/translation.json' --save-dir ./results
```

To enable BLEU and ROUGE scores as evaluation metrics, use the following command:
```bash
python main.py --data-path './data/translation.json' --save-dir ./results --enable-bleu-rouge
```

## Interpreting the Evaluation Results
The results are provided in two sections: individual_results and overall_results.

1. individual_results
    - This section contains the evaluation results for each instance.
    - It includes information about the input, predicted label, and scores (e.g., confidence, BLEU).
2. overall_results
    - This section provides aggregated metrics for the entire dataset.
    - Key metrics include accuracy, precision, recall, F1 score, and average confidence score.
    - For tasks evaluating the translation quality classifier, BLEU and ROUGE scores can also be included if enabled.

### Example Interpretation
If the LLM evaluator predicts the same label as the given instance with high confidence (e.g., above 0.5), it indicates that the model being evaluated has correctly classified the instance.

If all metrics in the overall_results are high (e.g., above 0.5), it suggests that the quality of the classifier is good.