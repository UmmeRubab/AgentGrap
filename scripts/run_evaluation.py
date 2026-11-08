import argparse
from pathlib import Path

from dotenv import load_dotenv
from agentgraph.dataset import load_examples
from agentgraph.evaluation import evaluate
from agentgraph.llm import GeminiLLM


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/hotpotqa/validation.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
    )
    parser.add_argument(
        "--backend",
        choices=["baseline", "gemini"],
        default="baseline",
    )

    args = parser.parse_args()

    examples = load_examples(
        args.data,
        args.limit,
    )

    llm = (
        GeminiLLM()
        if args.backend == "gemini"
        else None
    )

    results = evaluate(
        examples,
        mode=args.mode,
        llm=llm,
    )

    output = Path("results")
    output.mkdir(exist_ok=True)

    filename = output / (
        f"{args.backend}_{args.mode}.csv"
    )

    results.to_csv(filename, index=False)

    print(results[
        [
            "exact_match",
            "answer_f1",
            "evidence_f1",
            "latency_seconds",
        ]
    ].mean())

    print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
