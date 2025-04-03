# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv sync
source .venv/bin/activate

# Run the main script to produce figures
uv run main