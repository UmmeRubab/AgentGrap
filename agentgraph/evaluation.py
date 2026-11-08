import pandas as pd

from .agents import run_agents
from .metrics import (
    exact_match,
    answer_f1,
    evidence_f1,
)


def evaluate(
    examples,
    mode="single",
    llm=None,
    top_k=6,
):
    results = []

    for example in examples:
        output = run_agents(
            example,
            mode=mode,
            llm=llm,
            top_k=top_k,
        )

        results.append({
            "id": example["id"],
            "question": example["question"],
            "reference": example["answer"],
            "prediction": output["answer"],
            "exact_match": exact_match(
                output["answer"],
                example["answer"],
            ),
            "answer_f1": answer_f1(
                output["answer"],
                example["answer"],
            ),
            "evidence_f1": evidence_f1(
                output["evidence"],
                example["supporting_facts"],
            ),
            "latency_seconds": output["latency"],
            "mode": mode,
            "backend": (
                "gemini" if llm else "lexical_baseline"
            ),
        })

    return pd.DataFrame(results)
