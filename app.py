from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agentgraph.dataset import (
    SAMPLE_PATH,
    load_examples,
)
from agentgraph.agents import run_agents
from agentgraph.llm import GeminiLLM
from agentgraph.metrics import (
    exact_match,
    answer_f1,
    evidence_f1,
)
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
    "The bundled sample questions are illustrative. "
    "Download official HotpotQA validation data "
    "before reporting benchmark results."
)

with st.sidebar:
    st.header("Configuration")

    dataset_choice = st.selectbox(
        "Dataset",
        [
            "Illustrative examples",
            "HotpotQA validation",
        ],
    )

    backend = st.selectbox(
        "Inference backend",
        [
            "Lexical baseline (no API)",
            "Gemini API",
        ],
    )

    mode = st.radio(
        "Workflow",
        ["single", "multi"],
    )

    top_k = st.slider(
        "Retrieved evidence sentences",
        2,
        12,
        6,
    )

    evaluation_limit = st.slider(
        "Evaluation examples",
        1,
        50,
        10,
    )

path = (
    SAMPLE_PATH
    if dataset_choice == "Illustrative examples"
    else Path("data/hotpotqa/validation.jsonl")
)

if not path.exists():
    st.warning(
        "HotpotQA has not been downloaded. "
        "Run: python -m scripts.download_dataset "
        "--limit 1000"
    )
    st.stop()

examples = load_examples(path, limit=100)

llm = None

if backend == "Gemini API":
    try:
        llm = GeminiLLM()
    except ValueError as error:
        st.error(str(error))
        st.stop()

tab1, tab2, tab3 = st.tabs([
    "Agent Playground",
    "Research Evaluation",
    "Architecture",
])

with tab1:
    st.subheader("Multi-Hop Question Answering")

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

    if st.button(
        "Run AgentGraph",
        type="primary",
    ):
        try:
            with st.spinner(
                "Executing workflow..."
            ):
                result = run_agents(
                    example,
                    mode=mode,
                    llm=llm,
                    top_k=top_k,
                )

            st.success(result["answer"])

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Exact Match",
                f"{exact_match(result['answer'], example['answer']):.0%}",
            )

            col2.metric(
                "Answer F1",
                f"{answer_f1(result['answer'], example['answer']):.2f}",
            )

            col3.metric(
                "Latency",
                f"{result['latency']:.2f}s",
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
                pd.DataFrame(result["evidence"]),
                use_container_width=True,
            )

            st.metric(
                "Supporting Evidence F1",
                f"{evidence_f1(result['evidence'], example['supporting_facts']):.2f}",
            )

        except Exception as error:
            st.error(f"Execution failed: {error}")

with tab2:
    st.subheader("Benchmark Evaluation")

    st.write(
        "Evaluate answer accuracy, evidence retrieval "
        "and latency on the selected dataset."
    )

    if st.button("Run Evaluation"):
        try:
            subset = examples[
                :evaluation_limit
            ]

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
                    metric.replace("_", " ").title(),
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
            st.error(f"Evaluation failed: {error}")

with tab3:
    st.subheader("Research Architecture")

    st.markdown("""
### Single-Agent Workflow

Question → Evidence Retrieval → QA Agent → Answer

### Multi-Agent Workflow

Question → Evidence Retrieval → Reasoning Agent
→ Verification Agent → Answer Agent → Final Answer

### Evaluation

- Exact Match
- Answer F1
- Supporting Evidence F1
- Inference Latency

### Important limitation

The no-API mode uses lexical retrieval and sentence
extraction. It does not perform LLM-based reasoning.

The Gemini multi-agent mode executes separate reasoning,
verification, and answer-generation calls.
""")
