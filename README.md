# AgentGraph

Graph-inspired Multi-Agent Reasoning for
Evidence-Grounded Question Answering.

## Technology

- Python
- Streamlit
- HotpotQA
- Gemini API (optional)
- Pandas
- Plotly
- Pytest

No Node.js or React required.

## Windows setup

Run:

```bat
setup_windows.bat
```

This creates `.venv` and installs dependencies.

Then run:

```bat
run_windows.bat
```

## macOS / Linux

```bash
bash setup_unix.sh
bash run_unix.sh
```

## Download HotpotQA

With the virtual environment active:

```bash
python -m scripts.download_dataset --limit 1000
```

The official dataset is downloaded from Hugging Face.

Dataset:
https://huggingface.co/datasets/hotpotqa/hotpot_qa

If downloading fails due to dataset compatibility or
network restrictions, check the dataset repository
and the installed `datasets` version.

## Optional Gemini API

Copy `.env.example` to `.env`.

Set:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

The default model may need to be updated to a model
currently available to your Gemini account.

Never commit your `.env` file.

API usage may incur charges.

## Run tests

```bash
python -m pytest -q
```

## Run evaluations

```bash
python -m scripts.run_evaluation --mode single --backend baseline
python -m scripts.run_evaluation --mode single --backend gemini
python -m scripts.run_evaluation --mode multi --backend gemini
```

Results are saved in `results/`.

## GitHub

Create an empty repository called `agentgraph`.

Then run:

```bash
git init
git branch -M main
git config user.name "YOUR NAME"
git config user.email "YOUR VERIFIED GITHUB EMAIL"
git add .
git commit -m "Initial AgentGraph implementation"
git remote add origin https://github.com/YOUR_USERNAME/agentgraph.git
git push -u origin main
```

Replace the placeholders.

## Scientific integrity

This repository provides a runnable research prototype,
not a completed or validated research publication.

Run the actual experiments before making performance
claims on your CV.

Document AI assistance where required by universities.
