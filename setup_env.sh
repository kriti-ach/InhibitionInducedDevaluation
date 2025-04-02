curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
source .venv/bin/activate

python -m inhibition_induced_devaluation.main