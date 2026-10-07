import os
import time
import yaml
import click
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from data_loader import load_dataset
from llm_utils import generate_groq, generate_vllm
from metrics import calculate_metrics, judge_answer
from rag_utils import build_hybrid_retriever, format_docs, get_rag_chain

load_dotenv()


def load_configs(config_dir: str = "configs") -> tuple[dict, dict, dict]:
    with open(os.path.join(config_dir, "data.yaml"), "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "models.yaml"), "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "rag.yaml"), "r", encoding="utf-8") as f:
        rag_cfg = yaml.safe_load(f)
    return data_cfg, models_cfg, rag_cfg


@click.group()
def cli():
    """CLI интерфейс RAG и Бейзлайнов."""
    pass


@cli.command()
@click.option("--config-dir", default="configs", help="Папка с YAML конфигами.")
@click.option("--generator", type=click.Choice(["groq", "vllm"]), required=True, help="Генератор ответа.")
def run_baseline(config_dir: str, generator: str):
    """Генерация ответов без RAG (Baseline)."""
    data_cfg, models_cfg, _ = load_configs(config_dir)
    out_dir = data_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    _, test_df = load_dataset(
        data_cfg["articles_path"],
        data_cfg["questions_path"],
        data_cfg["ground_truth_path"]
    )

    click.echo(f"Запуск генерации baseline ({generator})...")
    gen_func = generate_groq if generator == "groq" else generate_vllm
    model_params = models_cfg[generator]
    prefix = "baseline_groq_qwen" if generator == "groq" else "baseline_llama_3b"

    answers = []
    for q in tqdm(test_df["question"]):
        answers.append(gen_func(q, **model_params))

    res_df = test_df[["id", "question", "answer"]].copy()
    res_df["baseline_answer"] = answers

    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    click.echo(f"Сохранено в {csv_path}")


@cli.command()
@click.option("--config-dir", default="configs", help="Папка с YAML конфигами.")
@click.option("--generator", type=click.Choice(["groq", "vllm"]), required=True, help="Генератор ответа")
def run_rag(config_dir: str, generator: str):
    """Генерация ответов с RAG"""
    data_cfg, models_cfg, rag_cfg = load_configs(config_dir)
    out_dir = data_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    articles, test_df = load_dataset(
        data_cfg["articles_path"],
        data_cfg["questions_path"],
        data_cfg["ground_truth_path"]
    )

    click.echo("Построение Hybrid Retriever...")
    retriever = build_hybrid_retriever(articles, rag_cfg)
    chain = get_rag_chain(generator, models_cfg[generator], rag_cfg["prompt"])

    rag_results = []
    click.echo(f"Генерация RAG ответов ({generator})...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q = row["question"]
        docs = retriever.invoke(q)[:rag_cfg["top_k_context"]]
        context = format_docs(docs)
        ans = chain.invoke({"context": context, "question": q})

        rag_results.append({
            "id": row["id"],
            "question": q,
            "answer": row["answer"],
            "rag_answer": ans,
            "retrieved_context": context
        })

    out_df = pd.DataFrame(rag_results)
    file_name = "rag_qwen_results.csv" if generator == "groq" else "rag_llama_3b_vllm_results.csv"
    save_path = os.path.join(out_dir, file_name)
    out_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    click.echo(f"Результаты RAG сохранены в {save_path}")


@cli.command()
@click.option("--config-dir", default="configs", help="Папка с YAML конфигами.")
@click.option("--target", type=click.Choice(["baseline", "rag"]), required=True, help="Тип пайплайна.")
def eval_classical(config_dir: str, target: str):
    """Расчет стандартных метрик BLEU и ROUGE."""
    data_cfg, _, _ = load_configs(config_dir)
    out_dir = data_cfg["output_dir"]

    if target == "baseline":
        groq_file, llama_file = "baseline_groq_qwen.csv", "baseline_llama_3b.csv"
        col, out_name = "baseline_answer", "baseline_metrics.csv"
    else:
        groq_file, llama_file = "rag_qwen_results.csv", "rag_llama_3b_vllm_results.csv"
        col, out_name = "rag_answer", "RAG_metrics.csv"

    results = []
    for model_title, file in [("Qwen 3.8 27B (Groq)", groq_file), ("Llama 3.2 3B Instruct", llama_file)]:
        path = os.path.join(out_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            preds = df[col].fillna("").tolist()
            refs = df["answer"].fillna("").tolist()
            m = calculate_metrics(preds, refs)
            results.append({"Model": model_title, **m})

    metrics_df = pd.DataFrame(results)
    save_path = os.path.join(out_dir, out_name)
    metrics_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    click.echo(f"Метрики рассчитаны:\n{metrics_df}\nСохранено в {save_path}")


@cli.command()
@click.option("--config-dir", default="configs", help="Папка с YAML конфигами.")
@click.option("--target", type=click.Choice(["baseline", "rag"]), required=True, help="Тип пайплайна.")
def eval_judge(config_dir: str, target: str):
    """Оценка ответов с помощью LLM-as-a-Judge."""
    data_cfg, models_cfg, _ = load_configs(config_dir)
    out_dir = data_cfg["output_dir"]
    judge_cfg = models_cfg["judge"]

    if target == "baseline":
        targets = [
            ("Qwen 3.8 27B (Groq)", "baseline_groq_qwen.csv", "baseline_answer", "baseline_groq_qwen_judge.csv"),
            ("Llama 3.2 3B Instruct", "baseline_llama_3b.csv", "baseline_answer", "baseline_llama_3b_judge.csv")
        ]
        out_summary = "baseline_judge_metrics.csv"
    else:
        targets = [
            ("Qwen 3.8 27B (Groq)", "rag_qwen_results.csv", "rag_answer", "rag_groq_qwen_judge.csv"),
            ("Llama 3.2 3B Instruct", "rag_llama_3b_vllm_results.csv", "rag_answer", "rag_llama_3b_judge.csv")
        ]
        out_summary = "rag_judge_metrics.csv"

    summary_data = []
    for model_name, in_file, ans_col, out_file in targets:
        in_path = os.path.join(out_dir, in_file)
        if not os.path.exists(in_path):
            continue

        df = pd.read_csv(in_path)
        scores = []
        click.echo(f"Оценка {model_name}...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sc = judge_answer(row["question"], row[ans_col], row["answer"], judge_cfg)
            scores.append(sc)
            time.sleep(0.2)

        df["judge_score"] = scores
        df.to_csv(os.path.join(out_dir, out_file), index=False, encoding="utf-8-sig")

        valid = [s for s in scores if s is not None]
        summary_data.append({
            ("Model" if target == "baseline" else "RAG with"): model_name,
            "LLM-as-a-Judge": sum(valid) / len(valid) if valid else 0
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(out_dir, out_summary)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    click.echo(f"Итоговые оценки LLM-as-a-judge:\n{summary_df}\nСохранено в {summary_path}")


if __name__ == "__main__":
    cli()