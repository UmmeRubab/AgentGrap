
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agentgraph.dataset import SAMPLE_PATH, load_examples
from agentgraph.agents import run_agents
from agentgraph.llm import GeminiLLM
from agentgraph.metrics import exact_match, answer_f1, evidence_f1
from agentgraph.evaluation import evaluate


load_dotenv()

st.set_page_config(
    page_title="AgentGraph",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AgentGraph")
st.caption(
    "Graph-inspired multi-agent reasoning "
    "and evidence-grounded question answering"
)

st.info(
    "Sample questions are illustrative. "
    "Use official HotpotQA validation data "
    "for benchmark evaluation."
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("Configuration")

    dataset_choice = st.selectbox(
        "Dataset",
        [
            "Illustrative examples",
            "HotpotQA validation",
        ],
    )

    mode = st.radio(
        "Workflow",
        ["single", "multi"],
    )

    top_k = st.slider(
        "Retrieved evidence sentences",
        min_value=2,
        max_value=12,
        value=6,
    )

    evaluation_limit = st.slider(
        "Evaluation examples",
        min_value=1,
        max_value=50,
        value=10,
    )


# -----------------------------
# Load dataset
# -----------------------------

path = (
    SAMPLE_PATH
    if dataset_choice == "Illustrative examples"
    else Path("data/hotpotqa/validation.jsonl")
)

examples = []

if path.exists():
    examples = load_examples(path, limit=100)
else:
    st.warning(
        "Selected dataset is not available. "
        "To download HotpotQA, run: "
        "python -m scripts.download_dataset --limit 1000"
    )


# -----------------------------
# Gemini
# -----------------------------

try:
    llm = GeminiLLM()
except ValueError as error:
    st.error(str(error))
    st.stop()


# -----------------------------
# Custom evidence helper
# -----------------------------

def build_custom_example(question, context):
    """
    Convert user-provided evidence into the
    same structure used by AgentGraph datasets.

    Each non-empty line becomes one evidence
    sentence. If a line contains multiple
    sentences, they are split approximately
    using punctuation boundaries.
    """

    sentences = []

    for line in context.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+",
            line,
        )

        sentences.extend(
            part.strip()
            for part in parts
            if part.strip()
        )

    return {
        "id": "custom-question",
        "question": question.strip(),
        "answer": "",
        "context": [
            {
                "title": "User-provided evidence",
                "sentences": sentences,
            }
        ],
        "supporting_facts": [],
    }


# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Agent Playground",
        "Research Evaluation",
        "Architecture",
    ]
)


# -----------------------------
# Agent Playground
# -----------------------------

with tab1:
    st.subheader("Multi-Hop Question Answering")

    question_source = st.radio(
        "Question source",
        [
            "Sample question",
            "Ask your own question",
        ],
        horizontal=True,
    )

    example = None
    is_custom = (
        question_source == "Ask your own question"
    )

    if is_custom:
        custom_question = st.text_input(
            "Enter your question",
            placeholder=(
                "e.g. Which university did "
                "the author of The Hobbit attend?"
            ),
        )

        custom_context = st.text_area(
            "Supporting evidence / context",
            placeholder=(
                "The Hobbit was written by "
                "J. R. R. Tolkien.\n"
                "J. R. R. Tolkien attended "
                "the University of Oxford."
            ),
            height=180,
            help=(
                "Paste source information relevant "
                "to your question. AgentGraph uses "
                "this evidence to produce an answer."
            ),
        )

        st.caption(
            "Custom questions require supporting "
            "evidence. Without it, AgentGraph cannot "
            "perform evidence-grounded reasoning."
        )

        if (
            custom_question.strip()
            and custom_context.strip()
        ):
            example = build_custom_example(
                custom_question,
                custom_context,
            )

    else:
        if examples:
            selected = st.selectbox(
                "Select a question",
                range(len(examples)),
                format_func=lambda index: (
                    examples[index]["question"]
                ),
            )

            example = examples[selected]

            st.write(
                "**Reference answer:**",
                example["answer"],
            )
        else:
            st.warning(
                "No sample questions are available "
                "for the selected dataset."
            )

    if st.button(
        "Run AgentGraph",
        type="primary",
    ):
        if example is None:
            st.warning(
                "Please provide both a question "
                "and supporting evidence."
            )
        else:
            try:
                with st.spinner(
                    "Executing AgentGraph workflow..."
                ):
                    result = run_agents(
                        example,
                        mode=mode,
                        llm=llm,
                        top_k=top_k,
                    )

                st.success(result["answer"])

                col1, col2, col3 = st.columns(3)

                if is_custom:
                    col1.metric(
                        "Exact Match",
                        "N/A",
                    )

                    col2.metric(
                        "Answer F1",
                        "N/A",
                    )

                else:
                    col1.metric(
                        "Exact Match",
                        (
                            f"{exact_match(result['answer'], example['answer']):.0%}"
                        ),
                    )

                    col2.metric(
                        "Answer F1",
                        (
                            f"{answer_f1(result['answer'], example['answer']):.2f}"
                        ),
                    )

                col3.metric(
                    "Latency",
                    f"{result['latency']:.2f}s",
                )

                if is_custom:
                    st.caption(
                        "Accuracy metrics are unavailable "
                        "because no reference answer was "
                        "provided for this custom question."
                    )

                st.subheader("Agent Execution Trace")

                for step in result["trace"]:
                    with st.expander(
                        step["agent"],
                        expanded=True,
                    ):
                        st.write(step["output"])

                st.subheader("Retrieved Evidence")

                st.dataframe(
                    pd.DataFrame(
                        result["evidence"]
                    ),
                    use_container_width=True,
                )

                if not is_custom:
                    st.metric(
                        "Supporting Evidence F1",
                        (
                            f"{evidence_f1(result['evidence'], example['supporting_facts']):.2f}"
                        ),
                    )
                else:
                    st.caption(
                        "Supporting Evidence F1 is not "
                        "available without annotated "
                        "ground-truth supporting facts."
                    )

            except Exception as error:
                st.error(
                    f"Execution failed: {error}"
                )


# -----------------------------
# Research Evaluation
# -----------------------------

with tab2:
    st.subheader("Benchmark Evaluation")

    st.write(
        "Evaluate answer accuracy, evidence "
        "retrieval and latency using the "
        "selected dataset."
    )

    if st.button("Run Evaluation"):
        if not examples:
            st.warning(
                "No dataset examples are available."
            )
        else:
            try:
                subset = examples[:evaluation_limit]

                with st.spinner(
                    "Evaluating questions..."
                ):
                    results = evaluate(
                        subset,
                        mode=mode,
                        llm=llm,
                        top_k=top_k,
                    )

                st.dataframe(
                    results,
                    use_container_width=True,
                )

                metrics = [
                    "exact_match",
                    "answer_f1",
                    "evidence_f1",
                    "latency_seconds",
                ]

                columns = st.columns(4)

                for column, metric in zip(
                    columns,
                    metrics,
                ):
                    column.metric(
                        metric.replace(
                            "_", " "
                        ).title(),
                        f"{results[metric].mean():.3f}",
                    )

                fig = px.bar(
                    results,
                    x="id",
                    y=[
                        "exact_match",
                        "answer_f1",
                        "evidence_f1",
                    ],
                    barmode="group",
                    title="Per-Question Evaluation",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.download_button(
                    "Download Results CSV",
                    results.to_csv(index=False),
                    file_name="agentgraph_results.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error(
                    f"Evaluation failed: {error}"
                )


# -----------------------------
# Architecture
# -----------------------------

with tab3:
    st.subheader("Research Architecture")

    st.markdown(
        """
### Single-Agent Workflow

Question → Evidence Retrieval → QA Agent → Answer

### Multi-Agent Workflow

Question → Evidence Retrieval → Reasoning Agent
→ Verification Agent → Answer Agent → Final Answer

### Custom Question Mode

User Question + Supporting Evidence
→ Retrieval → Agent Workflow → Answer

### Evaluation Metrics

- Exact Match
- Answer F1
- Supporting Evidence F1
- Inference Latency

### Research Notes

- Gemini powers the reasoning workflow.
- Custom questions use user-provided evidence.
- Benchmark scores require reference answers.
- Supporting Evidence F1 requires annotated facts.
- Official HotpotQA data should be used for
  research-quality evaluation.
"""
    )
