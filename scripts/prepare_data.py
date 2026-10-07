import json
import pandas as pd


def load_dataset(articles_path: str, questions_path: str, ground_truth_path: str):
    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    questions_df = pd.DataFrame(questions)
    ground_truth_df = pd.DataFrame(ground_truth)
    test_df = questions_df.merge(ground_truth_df, on="id")

    return articles, test_df