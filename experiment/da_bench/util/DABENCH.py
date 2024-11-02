import os
import fire
import sys
import json

import yaml
from openai import OpenAI

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import nest_asyncio

from experiment.da_bench.util.exceptions import handle_exception

DATA_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, "data")
DABENCH_PROMPT = "You are required to {question} from a CSV file named {file_name}. **Constraints**: Ensure that {constraints}, which must be strictly followed throughout the task.  The output format should be {format}. This task is categorized as {level}."
DABENCH_PATH = os.path.join(DATA_PATH, "da_bench")


def evaluate_accuracy_by_question(results: list[dict]) -> float:
    """
    Calculate the accuracy of results based on complete correctness of each question.
    This function is referenced from https://github.com/InfiAgent/InfiAgent/blob/main/examples/DA-Agent/eval_closed_form.py
    This function checks whether each result is entirely correct, meaning all sub-questions
    within that result are answered correctly. It computes the proportion of correct results
    by dividing the number of fully correct results by the total number of results.

    Args:
        results (dict): A collection of results where each result may contain a 'correctness' field.

    Returns:
        float: The proportion of correct results, rounded to four decimal places.
               Returns 0 if there are no results.
    """
    correct = sum("correctness" in result and all(result["correctness"].values()) for result in results)
    total = len(results)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_by_sub_question(results: list[dict]) -> float:
    """
    Evaluate the correctness of all sub-questions across the results.
    This function is referenced from https://github.com/InfiAgent/InfiAgent/blob/main/examples/DA-Agent/eval_closed_form.py
    This function calculates the total number of correct sub-questions and the overall
    number of sub-questions present in all results. It returns the ratio of correct
    sub-questions to the total number of sub-questions.

    Args:
        results (dict): A collection of results where each result may contain a 'correctness' field.

    Returns:
        float: The ratio of correct sub-questions, rounded to four decimal places.
               Returns 0 if there are no sub-questions.
    """
    correct = sum(sum(result["correctness"].values()) for result in results if "correctness" in result)
    total = sum(len(result["correctness"]) for result in results if "correctness" in result)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_proportional_by_sub_question_adjusted(results: list[dict]) -> float:
    """
    Adjust the score based on the number of sub-questions in each result.
    This function is referenced from https://github.com/InfiAgent/InfiAgent/blob/main/examples/DA-Agent/eval_closed_form.py
    This function calculates a score for each result by considering the number of sub-questions
    it contains. Each sub-question is assigned a score of 1 divided by the number of sub-questions.
    The total score for each result is computed as the sum of all correct sub-questions multiplied
    by the score per sub-question. Finally, it returns the average score across all results.

    Args:
        results (dict): A collection of results where each result may contain a 'correctness' field.

    Returns:
        float: The average score across all results, rounded to four decimal places.
               Returns 0 if there are no results.
    """
    total_score = 0
    for result in results:
        if "correctness" in result:
            sub_question_count = len(result["correctness"])
            score_per_sub_question = 1 / sub_question_count if sub_question_count > 0 else 0
            question_score = sum(result["correctness"].values()) * score_per_sub_question
            total_score += question_score
    return round(total_score / len(results), 4) if results else 0


def evaluate_completeness_by_question(results: list[dict]) -> float:
    none_cnt = sum(result["correctness"] is None for result in results)
    return round(1 - none_cnt / len(results), 4)


async def reformat(question: str, format: str, response: str) -> str:
    """
    Asynchronously reformats a given response based on specified formatting requirements.
    This function is referenced from https://github.com/InfiAgent/InfiAgent/blob/main/examples/DA-Agent/reformat.py
    This function constructs a prompt for the LLM (Large Language Model) to reformat
    the provided response according to the specified format. It includes a system prompt
    to guide the LLM's behavior and a template that outlines the expected output structure.

    Args:
        question (str): The original question posed by the user.
        format (str): The specific formatting requirements that the response must adhere to.
        response (str): The initial response from the LLM that needs to be reformatted.

    Returns:
        str: The reformatted response generated by the LLM based on the provided question
             and formatting requirements.
    """
    system_prompt = "You are a helpful assistant."
    demons = """\Format{{
        @shapiro_wilk_statistic[test_statistic]
        @shapiro_wilk_p_value[p_value]
        where "test_statistic" is a number between 0 and 1 representing the Shapiro-Wilk test statistic. Rounding off the answer to two decimal places.
        where "p_value" is a number between 0 and 1 representing the p-value from the Shapiro-Wilk test. Rounding off the answer to four decimal places.
        }}
        \Answer{{
        @shapiro_wilk_statistic[0.56]
        @shapiro_wilk_p_value[0.0002]   
        }}

        \Format{{
        @total_votes_outliers_num[outlier_num]
        where "outlier_num" is an integer representing the number of values considered outliers in the 'total_votes' column.
        }}
        \Answer{{
        @total_votes_outliers[10]   
        }}
        """
    reformat_template = """You should strictly follow the output requirements in the Format part. Here're some examples: {demons}. 
    Your answer should contain all the \"@answer_name[answer]\" in the order mentioned, each \"answer\" should be in the range of value as required. You need to keep the original numbers and text, just reformat without making any changes.
    The format requirements of this question is:
    {format}. You need to keep the original numbers and text, just reformat without making any changes. Please give your answer:"""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
        {"role": "user", "content": reformat_template.format(demons=demons, format=format)},
    ]
    rsp = await ask(messages, system_prompt)
    return rsp


def load_jsonl(file_path: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file into a list of dictionaries.

    Args:
        file_path (Union[Path, str]): The path to the JSONL file to be loaded.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the data from the JSONL file.
    """
    # Convert file_path to Path if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def compare_predictions(pred_dict: dict, true_label: list) -> bool:
    """
    Compares each prediction against the corresponding true label.

    This function checks whether the predicted values match the true values for each
    metric. It sorts the true labels to ensure the comparison is made in the correct
    order. The function returns True if all predictions are accurate within a small
    tolerance for numerical values, or if string values match case-insensitively.

    Args:
        pred_dict (dict): A dictionary of predicted metrics and their values.
        true_label (list): A list of tuples containing true metrics and their values.

    Returns:
        bool: True if all predictions match the true labels, False otherwise.
    """
    sorted_true_label = sorted(true_label, key=lambda x: x[0])  # Sort true labels by metric name

    for metric, true_value in sorted_true_label:
        try:
            true_value = float(true_value)  # Attempt to convert the true value to float
        except ValueError:
            true_value = true_value.replace(",", "")  # Clean the true value if conversion fails

        # Check if the true value is numeric and compare with the prediction
        if isinstance(true_value, (int, float)) and (
            metric not in pred_dict or abs(pred_dict[metric] - true_value) > 1e-6
        ):
            return False  # Return False if the prediction is inaccurate

        # Check if the true value is a string and compare with the prediction
        if isinstance(true_value, str) and (
            metric not in pred_dict or str(pred_dict[metric]).lower() != str(true_value).lower()
        ):
            return False  # Return False if the string prediction does not match

    return True  # Return True if all predictions are accurate


async def ask(question: str, system_prompt: str) -> str:
    # FIXME
    pass
    # from metagpt.llm import LLM  # Importing the LLM class from the metagpt module
    #
    # gpt4o_config_path = METAGPT_ROOT / "config" / "gpt-4o.yaml"
    # gpt4o_config = Config.from_yaml_file(gpt4o_config_path)
    #
    # llm = LLM(llm_config=gpt4o_config.llm)
    # rsp = await llm.aask(question, system_msgs=[system_prompt])
    # return rsp


def parse_prediction(prediction: str) -> dict:
    """
    Parses a prediction string into a dictionary of metric-value pairs.

    This function takes a formatted string containing metrics and their corresponding
    values, separated by the "@" symbol. Each metric may be enclosed in brackets and
    may include commas. The function processes the input to extract and clean the
    metrics and their values, returning them in a structured dictionary format.

    Args:
        prediction (str): A string representation of metrics and their values.

    Returns:
        dict: A dictionary where each key is a metric name and each value is the
              corresponding value, either as a float or a string.
    """
    pred_dict = {}
    for pred in prediction.split("@"):
        if pred == "":
            continue  # Skip any empty segments resulting from the split
        temp = re.split(r"[\[\]]", pred.strip())  # Split the string by brackets
        temp = [s.replace(",", "") for s in temp]  # Remove commas from the segments
        parts = [s for s in temp if s]  # Filter out any empty strings

        if len(parts) < 2:
            continue

        metric = parts[0].strip().replace(",", "")  # Extract and clean the metric name
        value = parts[-1].replace(",", "").replace(":", "")  # Extract and clean the value

        try:
            value = float(value)  # Attempt to convert the value to a float
        except ValueError:
            pass  # If conversion fails, retain the value as a string

        pred_dict[metric] = value  # Store the metric-value pair in the dictionary
    return pred_dict


class DABench:
    def __init__(
        self,
        questions_file: Path = Path(DABENCH_PATH) / "da-dev-questions.jsonl",
        answers_file: Path = Path(DABENCH_PATH) / "da-dev-labels.jsonl",
        template: str = "",
    ):
        """
        Initializes the DABench instance with questions and answers.

        This constructor loads questions and answers from specified JSONL files.
        It also sets a template for formatting prompts. If no template is provided,
        a default template is used.

        Args:
            questions_file (Path): The path to the JSONL file containing questions.
            answers_file (Path): The path to the JSONL file containing answers.
            template (str): A string template for formatting prompts.
        """

        self.questions = {
            int(line["id"]): line for line in load_jsonl(questions_file)
        }  # Load questions from the specified file
        self.answers = {
            int(line["id"]): line for line in load_jsonl(answers_file)
        }  # Load answers from the specified file
        self.template = template if template else DABENCH_PROMPT  # Set the template, defaulting if necessary

    def get_question(self, question_id: str) -> dict:
        """
        Retrieve the question associated with the given ID.

        This method looks up a question by its unique identifier. If the question
        is found, it returns the question data; otherwise, it returns a message
        indicating that the question was not found.

        Args:
            question_id (str): The unique identifier for the question.

        Returns:
            dict: The question data if found, otherwise a "Question not found." message.
        """
        return self.questions.get(question_id, "Question not found.")  # Return the question or an error message

    def generate_formatted_prompt(self, question_id: str) -> str:
        """
        Generate a formatted prompt for the specified question ID.

        This method retrieves the question data and formats it using the specified
        template. The formatted prompt includes the question, constraints, format,
        file name, and level, allowing for a structured output.

        Args:
            question_id (str): The unique identifier for the question.

        Returns:
            str: A formatted prompt string based on the question data.
        """
        temp = self.get_question(question_id)  # Retrieve the question data
        return self.template.format(
            question=temp["question"],
            constraints=temp["constraints"],
            format=temp["format"],
            file_name=str(DABENCH_PATH) + "/da-dev-tables/" + temp["file_name"],
            level=temp["level"],
        )  # Format and return the prompt

    def get_answer(self, answer_id: str) -> list:
        """
        Retrieve the answer list associated with the given ID.

        This method looks up an answer by its unique identifier. If the answer
        is found, it returns the answer data; otherwise, it returns a message
        indicating that the answer was not found.

        Args:
            answer_id (str): The unique identifier for the answer.

        Returns:
            list: The answer data if found, otherwise an "Answer not found." message.
        """
        return self.answers.get(answer_id, "Answer not found.")  # Return the answer or an error message

    @handle_exception(exception_msg="Error parsing cleaned prediction", default_return=(None, False))
    def parse_cleaned_prediction(self, cleaned_prediction: str, true_label: Any) -> Tuple[str, bool]:
        """
        Parse the cleaned prediction and compare it with the true label.

        Args:
            cleaned_prediction (str): The cleaned prediction string.
            true_label (Any): The true label to compare against.

        Returns:
            Tuple[str, bool]: A tuple containing the cleaned prediction and a boolean indicating
                              whether it matches the true label.
        """
        if cleaned_prediction:  # Ensure the cleaned prediction is not empty
            pred_dict = parse_prediction(cleaned_prediction)  # Parse the prediction
            if pred_dict is not None and compare_predictions(pred_dict, true_label):
                return cleaned_prediction, True  # Return if the prediction matches the true label
        return cleaned_prediction, False  # Return the cleaned prediction with a False match

    @handle_exception(exception_msg="Error during async reformat", default_return=(None, False))
    def async_reformat_prediction(self, id: str, result: str) -> str:
        """
        Reformat the prediction asynchronously and extract the answer.

        Args:
            id (str): The identifier for the question.
            result (str): The original prediction result.

        Returns:
            str: The reformatted prediction or the original prediction if extraction fails.
        """
        question = self.get_question(id)["question"]  # Retrieve the question based on the ID
        question_format = self.get_question(id)["format"]  # Get the format of the question
        prediction = asyncio.run(reformat(question, question_format, result))  # Asynchronously reformat the prediction

        # Attempt to extract the answer from the reformatted prediction
        answer_part = prediction.split("Answer{{") if "Answer{{" in prediction else []
        if len(answer_part) > 1:
            return answer_part[1].split("}}")[0].strip()  # Return the extracted answer

        return prediction  # If extraction fails, return the original prediction

    def eval(self, id: str, result: str) -> Tuple[str, bool]:
        """
        Evaluate the prediction against the true label.

        Args:
            id (str): The identifier for the question.
            result (str): The original prediction result.

        Returns:
            Tuple[str, bool]: A tuple containing the final prediction and a boolean indicating
                              whether it matches the true label.
        """
        true_label = self.get_answer(id)["common_answers"]  # Retrieve the true label for comparison
        nest_asyncio.apply()  # Apply nested asyncio to allow for async calls
        result = json.loads(str(result).split("Current Plan")[1].split("## Current Task")[0])[-1]["result"].strip()
        cleaned_prediction = result.replace("{", "").replace("}", "").replace("'", "")  # Clean the prediction string

        # Use the decorated function to handle exceptions while parsing the cleaned prediction
        parsed_result = self.parse_cleaned_prediction(cleaned_prediction, true_label)
        if parsed_result[1]:  # If the parsed prediction is valid
            return parsed_result  # Return the valid prediction

        # If the cleaned prediction is not valid, attempt to asynchronously reformat it
        prediction = self.async_reformat_prediction(id, result)

        pred_dict = parse_prediction(prediction)  # Parse the reformatted prediction
        if pred_dict is not None and compare_predictions(pred_dict, true_label):
            return prediction, True  # Return if the reformatted prediction matches the true label

        return prediction, False  # Return the final prediction with a False match

    @handle_exception(exception_msg="Error evaluating single prediction", default_return={})
    def single_eval(self, id: str, prediction: str) -> dict:
        """
        Evaluate the prediction against the true label for a single question.
        just using in eval_all

        Args:
            id (str): The identifier for the question.
            prediction (str): The prediction string to evaluate.

        Returns:
            dict: A dictionary indicating the correctness of each metric.
        """
        true_label = self.get_answer(id)["common_answers"]  # Retrieve the true label for the question
        # Initialize the correctness dictionary with False values for each metric
        correctness = {metric: False for metric, _ in true_label}

        if prediction is None:
            return correctness

        prediction = prediction.replace("{", "").replace("}", "").replace("'", "")  # Clean the prediction string
        # nest_asyncio.apply()
        # prediction = self.async_reformat_prediction(id, prediction)
        pred_dict = parse_prediction(prediction)  # Parse the prediction into a dictionary

        # Check each metric's prediction against the true label
        for metric, true_value in true_label:
            try:
                true_value = float(true_value)  # Attempt to convert the true value to float
            except ValueError:
                true_value = true_value.replace(",", "")  # Handle non-numeric values

            if metric in pred_dict:
                # Consider the prediction correct if it's within a small tolerance
                if (
                    isinstance(true_value, (int, float))
                    and isinstance(pred_dict[metric], (int, float))
                    and abs(pred_dict[metric] - true_value) < 1e-6
                ):
                    correctness[metric] = True  # Mark as correct if within tolerance

                if isinstance(true_value, str) and (
                    metric not in pred_dict or str(pred_dict[metric]).lower() != str(true_value).lower()
                ):
                    correctness[metric] = True  # Mark as correct for string comparison

        return correctness  # Return the correctness dictionary

    def eval_all(self, id_list: list, predictions: list) -> dict:
        """
        Evaluate all predictions and calculate accuracy rates.

        Args:
            id_list (list): A list of question identifiers.
            predictions (list): A list of prediction strings corresponding to the questions.

        Returns:
            dict: A dictionary containing accuracy rates by question and sub-question.
        """
        results = []  # Initialize a list to store results for each question

        # Evaluate each prediction against its corresponding question ID
        for id, prediction in zip(id_list, predictions):
            correct = self.single_eval(id, prediction)
            results.append({"id": id, "correctness": correct})  # Append the result to the list

        # Calculate the three accuracy rates based on the results
        accuracy_by_question = evaluate_accuracy_by_question(results)
        accuracy_by_sub_question = evaluate_accuracy_by_sub_question(results)
        proportional_accuracy_by_sub_question = evaluate_accuracy_proportional_by_sub_question_adjusted(results)
        completeness_by_question = evaluate_completeness_by_question(results)

        return {
            "accuracy_by_question": accuracy_by_question,
            "accuracy_by_sub_question": accuracy_by_sub_question,
            "proportional_accuracy_by_sub_question": proportional_accuracy_by_sub_question,
            "completeness_by_question": completeness_by_question,
        }