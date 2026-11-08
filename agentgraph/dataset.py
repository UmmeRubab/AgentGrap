import json
from pathlib import Path

SAMPLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sample.json"
)


def normalize(record):
    context = record["context"]

    if isinstance(context, dict):
        context = [
            {"title": title, "sentences": sentences}
            for title, sentences in zip(
                context["title"], context["sentences"]
            )
        ]

    supporting = record.get("supporting_facts", [])

    if isinstance(supporting, dict):
        supporting = list(
            zip(
                supporting.get("title", []),
                supporting.get("sent_id", []),
            )
        )

    return {
        "id": str(record.get("id", record.get("_id", ""))),
        "question": record["question"],
        "answer": str(record["answer"]),
        "context": context,
        "supporting_facts": supporting,
    }


def load_examples(path=SAMPLE_PATH, limit=100):
    path = Path(path)

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as file:
            records = []
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
                if len(records) >= limit:
                    break
    else:
        with path.open(encoding="utf-8") as file:
            records = json.load(file)[:limit]

    return [normalize(item) for item in records]


def download_hotpotqa(
    limit=1000,
    output="data/hotpotqa/validation.jsonl",
):
    from datasets import load_dataset

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        for row in dataset.select(
            range(min(limit, len(dataset)))
        ):
            record = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "context": row["context"],
                "supporting_facts": row["supporting_facts"],
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    return destination
