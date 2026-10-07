import os
from groq import Groq
from openai import OpenAI


def generate_groq(question: str, model_name: str, max_tokens: int = 300, temperature: float = 0) -> str:
    """Генерация ответа через Groq API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Переменная окружения GROQ_API_KEY не установлена!")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort="none"
    )
    return response.choices[0].message.content.strip()


def generate_vllm(
    question: str,
    model_name: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    max_tokens: int = 300,
    temperature: float = 0
) -> str:
    """Генерация ответа через vLLM сервер."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()