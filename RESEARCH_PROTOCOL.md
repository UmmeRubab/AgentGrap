# AgentGraph Research Protocol

## Research question

Can a multi-agent reasoning and verification workflow
improve multi-hop question answering compared with
a single-agent LLM workflow?

## Dataset

Official HotpotQA distractor validation split.

https://hotpotqa.github.io/

## Experimental comparisons

1. Single-agent Gemini QA
2. Multi-agent Gemini QA with verification
3. Retrieval evidence quality
4. Latency comparison
5. Error analysis

## Metrics

- Exact Match
- Answer token F1
- Retrieved supporting-fact F1
- Mean inference latency

## Experimental controls

Use the same:
- Question IDs
- Retrieval configuration
- LLM model
- Model temperature
- Evaluation dataset

## Limitations

The lexical baseline is not a trained question-answering model.

The multi-agent workflow uses sequential LLM calls.
It does not train independent agents.

The evidence metric evaluates retrieved supporting facts,
not the model's explicit evidence predictions.

The bundled examples are illustrative and cannot support
claims about HotpotQA benchmark performance.

Do not report accuracy improvements without running
and comparing both Gemini configurations.
