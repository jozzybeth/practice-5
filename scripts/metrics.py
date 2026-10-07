import json
import sacrebleu
from rouge_score import rouge_scorer
from llm_utils import generate_groq


def calculate_metrics(predictions: list, references: list) -> dict:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge_scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]

    r1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / max(len(rouge_scores), 1)
    r2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / max(len(rouge_scores), 1)
    rL = sum(s["rougeL"].fmeasure for s in rouge_scores) / max(len(rouge_scores), 1)

    return {
        "BLEU": bleu.score,
        "ROUGE-1": r1,
        "ROUGE-2": r2,
        "ROUGE-L": rL,
    }


def judge_answer(question: str, answer: str, ground_truth: str, judge_config: dict) -> int | None:
    """Оценка ответа через LLM-as-a-judge"""
    prompt = judge_config["prompt"].format(
        question=question,
        answer=answer,
        ground_truth=ground_truth
    )

    llm_params = {k: v for k, v in judge_config.items() if k != "prompt"}

    try:
        response = generate_groq(prompt, **llm_params)
        result = json.loads(response)
        score = int(result["score"])
        return score if score in [0, 1, 2] else None
    except Exception as e:
        print("Ошибка ответа судьи:", e)
        return None