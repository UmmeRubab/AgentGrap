from agentgraph.dataset import load_examples
from agentgraph.agents import run_agents
from agentgraph.metrics import (
    exact_match,
    answer_f1,
    evidence_f1,
)
from agentgraph.retrieval import retrieve


def test_dataset():
    examples = load_examples()
    assert len(examples) == 2


def test_retrieval():
    example = load_examples()[0]
    evidence = retrieve(
        example["question"],
        example["context"],
    )
    assert len(evidence) > 0


def test_baseline():
    example = load_examples()[0]
    result = run_agents(example)
    assert result["answer"]
    assert result["trace"]


def test_exact_match():
    assert exact_match(
        "The Oxford",
        "Oxford",
    ) == 1.0


def test_answer_f1():
    assert answer_f1(
        "University of Oxford",
        "Oxford",
    ) > 0


def test_evidence_f1():
    evidence = [{
        "title": "Document",
        "index": 0,
        "text": "Example",
    }]

    assert evidence_f1(
        evidence,
        [("Document", 0)],
    ) == 1.0
