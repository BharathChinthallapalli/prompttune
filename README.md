# PromptTune: Fine-Tuning LLMs for Prompt Engineering

## Overview

This project demonstrates fine-tuning a Large Language Model (LLM) using Prompt Tuning (a PEFT technique) to improve user prompts. It also includes an LLM chain to analyze, suggest improvements, and synthesize better prompts.

## Features

- Fine-tunes LLMs (e.g., BLOOMz, Llama, Mistral) for prompt optimization tasks.
- Uses Prompt Tuning via the PEFT library for efficient adaptation.
- Processes various CSV formats for training data.
- Evaluates model performance using ROUGE and BERTScore.
- Includes an LLM chain (`llm_prompt_chain`) with three stages:
    1. Analyze prompt flaws.
    2. Recommend improvement techniques.
    3. Synthesize an improved prompt.

## Notebook: `finetune.ipynb`

### Purpose

Details the steps for data loading, preprocessing, model fine-tuning, evaluation, and using the prompt improvement chain.

### Key Components

-   **Setup and Dependencies:** Lists necessary Python libraries.
-   **Data Loading:** Explains how to upload CSV files and how the script processes them (`labeled_train_final.csv`, `prompt_examples_dataset.csv`, etc.). Briefly mention the `auto_col` helper.
-   **Preprocessing:** Describes tokenization and formatting for the model.
-   **Model Configuration:** Mentions the base model (e.g., `bigscience/bloomz-560m`) and PEFT `PromptTuningConfig`.
-   **Training:** Details the training process using `Trainer`.
-   **Inference:** Shows how to generate improved prompts with the fine-tuned model, including the `build_fewshot_prompt` helper.
-   **Evaluation:** Explains the use of ROUGE and BERTScore.
-   **LLM Prompt Improvement Chain:** Describes the `llm_analyze_flaws`, `llm_recommend_techniques`, `llm_synthesize_prompt`, and the main `llm_prompt_chain` functions.

## Setup and Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd prompttune
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: We'll need to create this `requirements.txt` file later based on the notebook's pip installs).*

## Data Preparation

-   The script expects CSV files for training and evaluation.
-   Examples of expected CSV structures or column names include:
    -   `original_prompt`, `improved_instruction`
    -   `bad_prompt`, `good_prompt`
    -   `task_description`, `prompting_techniques`
-   Place your CSV files in a specific directory (e.g., `./data`) or use the upload mechanism if running in Google Colab. The notebook demonstrates loading from specific paths.

## Running the Notebook

1.  Open `finetune.ipynb` in a Jupyter environment (like Jupyter Lab or Google Colab).
2.  Follow the cells sequentially to execute the code.
3.  Modify parameters in the configuration cells as needed (e.g., base model name, training hyperparameters, file paths).

## Example Usage (LLM Prompt Chain)

```python
# Assuming peft_model and tokenizer are loaded and configured
# If running locally, you might need to ensure finetune.py or relevant functions are in PYTHONPATH
# from finetune import llm_prompt_chain # Adjust import based on your setup

# Example (conceptual, assuming functions are defined and model loaded as in the notebook):
# user_prompt = "Tell me about dogs."
# improved_prompt = llm_prompt_chain(user_prompt, peft_model, tokenizer, verbose=True) # Ensure llm_prompt_chain is callable
# print(f"Original Prompt: {user_prompt}")
# print(f"Improved Prompt: {improved_prompt}")
```
*(Note: The exact method to import/call `llm_prompt_chain` will depend on how you structure the code from the notebook. The example above is illustrative.)*

## Future Enhancements (Optional)

-   Parameterize the choice of base LLM more dynamically.
-   Implement functionality to easily save and load fine-tuned PEFT adapters.
-   Develop a more interactive user interface (e.g., using Gradio or Streamlit) for easier prompt input and improvement.
-   Support for more PEFT techniques beyond Prompt Tuning.
-   Automated hyperparameter optimization.
```
