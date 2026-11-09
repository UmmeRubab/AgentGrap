
from pathlib import Path
import re
import html

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agentgraph.dataset import SAMPLE_PATH, load_examples
from agentgraph.agents import run_agents
from agentgraph.llm import GeminiLLM
from agentgraph.metrics import exact_match, answer_f1, evidence_f1
from agentgraph.evaluation import evaluate


# --------------------------------------------------
# Application setup
# --------------------------------------------------

load_dotenv()

NETWORK_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">
  <g fill="none" stroke="#6236E8" stroke-width="5"
     stroke-linecap="round" stroke-linejoin="round">
    <path d="M15 55 L30 17 L55 34 L65 61 L35 66 Z"/>
    <path d="M30 17 L35 66 M55 34 L35 66"/>
  </g>
  <g fill="#6236E8">
    <circle cx="15" cy="55" r="7"/>
    <circle cx="30" cy="17" r="7"/>
    <circle cx="55" cy="34" r="7"/>
    <circle cx="65" cy="61" r="7"/>
    <circle cx="35" cy="66" r="7"/>
  </g>
</svg>
"""

st.set_page_config(
    page_title="AgentGraph | Multi-Agent Research",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Visual theme
# --------------------------------------------------

st.markdown(
    """
<style>
    :root {
        --primary: #6236E8;
        --primary-dark: #4F25CB;
        --text: #17213D;
        --muted: #64708B;
        --border: #E8E5F4;
    }

    .stApp {
        background-color: #FFFFFF;
        color: var(--text);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #F7F5FF;
        border-right: 1px solid #E8E5F4;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #17213D;
    }

    h1, h2, h3 {
        color: #17213D !important;
        letter-spacing: -0.4px;
    }

    .agentgraph-hero {
        display: flex;
        align-items: center;
        gap: 20px;
        background: linear-gradient(
            115deg,
            #FAF9FF 0%,
            #F0ECFF 100%
        );
        padding: 25px 30px;
        border: 1px solid #EAE5FF;
        border-radius: 17px;
        margin-bottom: 18px;
    }

    .agentgraph-logo {
        width: 66px;
        height: 66px;
        flex-shrink: 0;
    }

    .agentgraph-title {
        font-size: 40px;
        font-weight: 750;
        letter-spacing: -1.5px;
        color: #17213D;
        line-height: 1.15;
    }

    .agentgraph-title span {
        color: #6236E8;
    }

    .agentgraph-subtitle {
        font-size: 15px;
        color: #64708B;
        margin-top: 7px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E8E5F4;
        margin-bottom: 18px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 10px 15px;
        color: #48536C;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: #F0EAFF;
        color: #6236E8 !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #6236E8;
        border: 1px solid #6236E8;
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #4F25CB;
        border-color: #4F25CB;
        color: white;
    }

    .stButton > button[kind="secondary"] {
        border-radius: 10px;
        border: 1px solid #DDD6FE;
    }

    div[data-testid="stMetric"] {
        background: #F8F6FF;
        border: 1px solid #EAE5FF;
        border-radius: 13px;
        padding: 17px;
    }

    div[data-testid="stMetricValue"] {
        color: #4F2DBD;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #E8E5F4;
        border-radius: 11px;
        background: white;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 11px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 11px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 9px;
    }

    .agentgraph-footer {
        text-align: center;
        color: #8B93A6;
        font-size: 12px;
        padding-top: 30px;
        padding-bottom: 8px;
    }

    @media (max-width: 768px) {
        .agentgraph-hero {
            padding: 20px;
            gap: 12px;
        }

        .agentgraph-logo {
            width: 45px;
            height: 45px;
        }

        .agentgraph-title {
            font-size: 30px;
        }

        .agentgraph-subtitle {
            font-size: 13px;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    f"""
<div class="agentgraph-hero">
    <div class="agentgraph-logo">
        {NETWORK_ICON}
    </div>
    <div>
        <div class="agentgraph-title">
            Agent<span>Graph</span>
        </div>
        <div class="agentgraph-subtitle">
            Multi-Agent Reasoning &amp; Evidence-Grounded QA
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "Illustrative examples are for demonstration. "
    "Use official HotpotQA validation data for benchmark evaluation."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    dataset_choice = st.selectbox(
        "Dataset",
        [
            "Illustrative examples",
            "HotpotQA validation",
        ],
    )

    workflow_label = st.radio(
        "Workflow",
        [
            "Single-Agent",
            "Multi-Agent",
        ],
    )

    mode = (
        "single"
        if workflow_label == "Single-Agent"
        else "multi"
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

    st.divider()

    st.caption(
        "AgentGraph is a research-oriented application "
        "for evaluating evidence-grounded question answering "
        "and multi-agent reasoning."
    )


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

path = (
    SAMPLE_PATH
    if dataset_choice == "Illustrative examples"
    else Path("data/hotpotqa/validation.jsonl")
)

examples = []

if path.exists():
    try:
        examples = load_examples(path, limit=100)
    except Exception as error:
        st.error(f"Dataset loading failed: {error}")
else:
    st.warning(
        "The selected dataset is not available. "
        "For HotpotQA, run: "
        "python -m scripts.download_dataset --limit 1000"
    )


# --------------------------------------------------
# Initialize Gemini
# --------------------------------------------------

try:
    llm = GeminiLLM()
except ValueError as error:
    st.error(str(error))
    st.stop()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def build_custom_example(question, context):
    """
    Convert user-provided evidence into the
    dataset structure expected by AgentGraph.
    """

    sentences = []

    for line in context.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = re.split(
            r"(?<=[.!?])\\s+",
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


def show_error(error, operation="Operation"):
    """
    Display a helpful message for common
    Gemini API and workflow errors.
    """

    message = str(error)

    if "503" in message:
        st.error(
            f"{operation} failed: Gemini is temporarily unavailable "
            "(HTTP 503). Try again later or reduce the number "
            "of evaluation examples."
        )

    elif "429" in message:
        st.error(
            f"{operation} failed: Gemini API rate limit or quota "
            "was reached (HTTP 429). Check your API usage."
        )

    else:
        st.error(f"{operation} failed: {message}")


def display_evaluation_metrics(results):
    """
    Display the average metrics from an evaluation.
    """

    metrics = [
        ("Exact Match", "exact_match"),
        ("Answer F1", "answer_f1"),
        ("Evidence F1", "evidence_f1"),
        ("Latency", "latency_seconds"),
    ]

    columns = st.columns(4)

    for column, (label, metric) in zip(columns, metrics):
        average = results[metric].mean()

        if metric == "latency_seconds":
            value = f"{average:.2f}s"
        else:
            value = f"{average:.3f}"

        column.metric(label, value)


# --------------------------------------------------
# Navigation
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Agent Playground",
        "Research Evaluation",
        "Architecture",
        "Workflow Comparison",
    ]
)


# ==================================================
# TAB 1 — AGENT PLAYGROUND
# ==================================================

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
                "Which university did the author "
                "of The Hobbit attend?"
            ),
        )

        custom_context = st.text_area(
            "Supporting evidence / context",
            placeholder=(
                "The Hobbit was written by J. R. R. Tolkien.\\n"
                "J. R. R. Tolkien attended the University of Oxford."
            ),
            height=150,
            help=(
                "Paste evidence relevant to your question. "
                "AgentGraph uses this evidence to generate an answer."
            ),
        )

        st.caption(
            "Custom questions require supporting evidence. "
            "Reference-based accuracy metrics are unavailable "
            "for questions without annotated answers."
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

            st.caption(
                f"Reference answer: {example['answer']}"
            )

        else:
            st.warning(
                "No sample questions are available."
            )

    if st.button(
        "Run AgentGraph",
        type="primary",
        key="run_agentgraph",
    ):

        if example is None:

            st.warning(
                "Please select a sample question or provide "
                "your own question and supporting evidence."
            )

        else:

            try:

                with st.spinner(
                    "Running evidence-grounded reasoning..."
                ):
                    result = run_agents(
                        example,
                        mode=mode,
                        llm=llm,
                        top_k=top_k,
                    )

                st.subheader("Generated Answer")

                st.success(result["answer"])

                col1, col2, col3, col4 = st.columns(4)

                if is_custom:

                    col1.metric(
                        "Exact Match",
                        "N/A",
                    )

                    col2.metric(
                        "Answer F1",
                        "N/A",
                    )

                    col3.metric(
                        "Evidence F1",
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
                        "Evidence F1",
                        (
                            f"{evidence_f1(result['evidence'], example['supporting_facts']):.2f}"
                        ),
                    )

                col4.metric(
                    "Latency",
                    f"{result['latency']:.2f}s",
                )

                if not is_custom:
                    st.caption(
                        "Exact Match is a strict text-based metric. "
                        "A factually correct answer with additional "
                        "words can receive a score of 0%."
                    )

                st.divider()

                left, right = st.columns(
                    [1, 1],
                    gap="large",
                )

                with left:

                    st.subheader("Agent Execution Trace")

                    for step in result["trace"]:

                        with st.expander(
                            step["agent"],
                            expanded=True,
                        ):
                            st.write(step["output"])

                with right:

                    st.subheader("Retrieved Evidence")

                    evidence_df = pd.DataFrame(
                        result["evidence"]
                    )

                    st.dataframe(
                        evidence_df,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as error:
                show_error(error, "Agent execution")


# ==================================================
# TAB 2 — RESEARCH EVALUATION
# ==================================================

with tab2:

    st.subheader("Research Evaluation")

    st.write(
        "Evaluate answer accuracy, supporting evidence "
        "retrieval, and inference latency."
    )

    if st.button(
        "Run Evaluation",
        type="primary",
        key="run_evaluation",
    ):

        if not examples:

            st.warning(
                "No dataset examples are available."
            )

        else:

            try:

                subset = examples[:evaluation_limit]

                with st.spinner(
                    "Evaluating selected questions..."
                ):
                    results = evaluate(
                        subset,
                        mode=mode,
                        llm=llm,
                        top_k=top_k,
                    )

                if results.empty:
                    st.warning(
                        "Evaluation returned no results."
                    )
                else:

                    st.subheader("Average Performance")

                    display_evaluation_metrics(results)

                    st.divider()

                    st.subheader("Per-Question Performance")

                    fig = px.bar(
                        results,
                        x="id",
                        y=[
                            "exact_match",
                            "answer_f1",
                            "evidence_f1",
                        ],
                        barmode="group",
                        title="Evaluation Metrics by Question",
                    )

                    fig.update_layout(
                        template="plotly_white",
                        legend_title_text="Metric",
                        margin=dict(
                            l=20,
                            r=20,
                            t=50,
                            b=20,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.subheader("Detailed Results")

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download Evaluation CSV",
                        data=results.to_csv(index=False),
                        file_name="agentgraph_evaluation.csv",
                        mime="text/csv",
                    )

                    if dataset_choice == "Illustrative examples":
                        st.info(
                            "These are illustrative examples, "
                            "not official HotpotQA benchmark scores."
                        )

            except Exception as error:
                show_error(error, "Evaluation")


# ==================================================
# TAB 3 — ARCHITECTURE
# ==================================================

with tab3:

    st.subheader("Research Architecture")

    st.write(
        "AgentGraph compares a single-agent question "
        "answering workflow with a sequential "
        "multi-agent reasoning and verification workflow."
    )

    st.markdown("#### Single-Agent Workflow")

    st.markdown(
        "Question → Evidence Retrieval → QA Agent → Answer"
    )

    st.divider()

    st.markdown("#### Multi-Agent Workflow")

    st.markdown(
        "Question → Evidence Retrieval → Reasoning Agent "
        "→ Verification Agent → Answer Agent → Final Answer"
    )

    st.divider()

    st.markdown("#### Evaluation Metrics")

    st.markdown(
        """
- **Exact Match:** Strict agreement with the reference answer.
- **Answer F1:** Token overlap between predicted and reference answers.
- **Evidence F1:** Retrieval overlap with annotated supporting facts.
- **Latency:** Time required to execute the workflow.
"""
    )

    st.divider()

    st.markdown("#### Research Notes")

    st.info(
        "The multi-agent workflow uses sequential Gemini "
        "calls for reasoning, verification, and final answering. "
        "It is not a graph execution engine or a system of "
        "independently trained agents."
    )

    st.write(
        "For a meaningful comparison, both workflows should "
        "use the same questions, model, retrieval settings, "
        "and evaluation criteria."
    )

    st.write(
        "The multi-agent workflow uses more model calls, "
        "so latency and API cost should be considered "
        "alongside answer quality."
    )

    st.caption(
        "Use official HotpotQA validation data for "
        "research-quality evaluation."
    )


# ==================================================
# TAB 4 — WORKFLOW COMPARISON
# ==================================================

with tab4:

    st.subheader(
        "Single-Agent vs Multi-Agent Comparison"
    )

    st.write(
        "Compare both workflows using identical questions, "
        "retrieval settings, and evaluation metrics."
    )

    comparison_limit = st.slider(
        "Comparison examples",
        min_value=1,
        max_value=min(10, len(examples)) if examples else 1,
        value=min(2, len(examples)) if examples else 1,
        key="comparison_limit",
    )

    st.caption(
        "Each comparison evaluates the same questions twice. "
        "The multi-agent workflow makes more Gemini API calls "
        "and may take longer."
    )

    if st.button(
        "Run Workflow Comparison",
        type="primary",
        key="run_comparison",
    ):

        if not examples:

            st.warning(
                "No dataset examples are available."
            )

        else:

            try:

                subset = examples[:comparison_limit]

                with st.spinner(
                    "Evaluating single-agent workflow..."
                ):
                    single_results = evaluate(
                        subset,
                        mode="single",
                        llm=llm,
                        top_k=top_k,
                    )

                with st.spinner(
                    "Evaluating multi-agent workflow..."
                ):
                    multi_results = evaluate(
                        subset,
                        mode="multi",
                        llm=llm,
                        top_k=top_k,
                    )

                single_results = single_results.copy()
                multi_results = multi_results.copy()

                single_results["workflow"] = "Single-Agent"
                multi_results["workflow"] = "Multi-Agent"

                combined = pd.concat(
                    [
                        single_results,
                        multi_results,
                    ],
                    ignore_index=True,
                )

                metrics = [
                    "exact_match",
                    "answer_f1",
                    "evidence_f1",
                    "latency_seconds",
                ]

                summary = (
                    combined.groupby(
                        "workflow",
                        as_index=False,
                    )[metrics]
                    .mean()
                )

                st.subheader("Performance Summary")

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                )

                single_avg = single_results[metrics].mean()
                multi_avg = multi_results[metrics].mean()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Answer F1 Difference",
                    (
                        f"{multi_avg['answer_f1'] - single_avg['answer_f1']:+.3f}"
                    ),
                )

                col2.metric(
                    "Exact Match Difference",
                    (
                        f"{multi_avg['exact_match'] - single_avg['exact_match']:+.3f}"
                    ),
                )

                col3.metric(
                    "Latency Difference",
                    (
                        f"{multi_avg['latency_seconds'] - single_avg['latency_seconds']:+.2f}s"
                    ),
                )

                st.divider()

                st.subheader("Answer Quality Comparison")

                chart_data = summary.melt(
                    id_vars="workflow",
                    value_vars=[
                        "exact_match",
                        "answer_f1",
                    ],
                    var_name="metric",
                    value_name="score",
                )

                fig = px.bar(
                    chart_data,
                    x="metric",
                    y="score",
                    color="workflow",
                    barmode="group",
                    title="Answer Quality by Workflow",
                )

                fig.update_layout(
                    template="plotly_white",
                    margin=dict(
                        l=20,
                        r=20,
                        t=50,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.subheader("Detailed Comparison")

                st.dataframe(
                    combined,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download Comparison CSV",
                    data=combined.to_csv(index=False),
                    file_name="agentgraph_comparison.csv",
                    mime="text/csv",
                )

                st.info(
                    "Evidence F1 measures retrieval quality, "
                    "not the verification agent's claim judgments. "
                    "Illustrative examples are not official "
                    "benchmark results."
                )

            except Exception as error:
                show_error(error, "Workflow comparison")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
<div class="agentgraph-footer">
    AgentGraph · Multi-Agent Reasoning Research Platform
</div>
""",
    unsafe_allow_html=True,
)
