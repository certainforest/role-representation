#!/usr/bin/env sh
set -eu
uv pip install \
  jupyterlab \
  pandas \
  tqdm \
  scikit-learn \
  openrouter \
  openai \
  python-dotenv \
  nnsight \
  torch

# Optional / pinned deps (uncomment if needed)
# uv pip install "transformers==4.48.3"
# uv pip install plotly pyyaml pyarrow termcolor datasets seaborn fastparquet wandb