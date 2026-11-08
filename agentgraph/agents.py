import time
from .retrieval import retrieve


def format_evidence(evidence):
    return "\n".join(
        f"[{row['title']} #{row['index']}] "
        f"{row['text']}"
        for row in evidence
    )


def run_agents(
    example,
    mode="multi",
    llm=None,
    top_k=6,
):
    if mode not in ("single", "multi"):
        raise ValueError("Invalid agent mode")

    start = time.perf_counter()

    evidence = retrieve(
        example["question"],
        example["context"],
        top_k,
    )

    context = format_evidence(evidence)
    question = example["question"]

    trace = [{
        "agent": "Retrieval",
        "output": f"Retrieved {len(evidence)} sentences.",
    }]

    if llm is None:
        # Transparent demonstration baseline.
        # This is NOT generative multi-agent reasoning.
        answer = (
            evidence[0]["text"]
            if evidence
            else "Insufficient evidence"
        )

        trace.append({
            "agent": "Lexical baseline",
            "output": "Selected the top evidence sentence.",
        })

    elif mode == "single":
        answer = llm.generate(
            "Answer using only the supplied evidence. "
            "Return a concise answer.",
            f"Question: {question}\n"
            f"Evidence:\n{context}",
        )

        trace.append({
            "agent": "Single QA Agent",
            "output": answer,
        })

    else:
        draft = llm.generate(
            "You are a research reasoning agent. "
            "Connect evidence across multiple documents. "
            "Do not invent facts.",
            f"Question: {question}\n"
            f"Evidence:\n{context}",
        )

        trace.append({
            "agent": "Reasoning Agent",
            "output": draft,
        })

        verification = llm.generate(
            "You are an evidence verification agent. "
            "Check the draft for unsupported claims "
            "and reasoning mistakes.",
            f"Question: {question}\n"
            f"Evidence:\n{context}\n"
            f"Draft:\n{draft}",
        )

        trace.append({
            "agent": "Verification Agent",
            "output": verification,
        })

        answer = llm.generate(
            "You are the final answer agent. "
            "Provide a short factual answer using "
            "the evidence and verification. "
            "If evidence is insufficient, say so.",
            f"Question: {question}\n"
            f"Evidence:\n{context}\n"
            f"Draft:\n{draft}\n"
            f"Verification:\n{verification}",
        )

        trace.append({
            "agent": "Answer Agent",
            "output": answer,
        })

    return {
        "answer": answer.strip(),
        "evidence": evidence,
        "trace": trace,
        "latency": time.perf_counter() - start,
        "mode": mode,
    }
