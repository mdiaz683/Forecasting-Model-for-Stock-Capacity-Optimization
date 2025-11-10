.PHONY: help setup install clean lint format test run

# Default target - show help
help:
	@echo "Available commands:"
	@echo "  setup    - Create virtual environment and install dependencies"
	@echo "  install  - Install/update dependencies only"
	@echo "  clean    - Remove virtual environment and cache files"
	@echo "  lint     - Check code quality with ruff and black"
	@echo "  format   - Auto-format code with black and ruff"
	@echo "  test     - Run tests with pytest"
	@echo "  run      - Start Streamlit application"

# Create virtual environment and install everything
setup:
	@echo "Creating virtual environment..."
	python -m venv .venv
	@echo "Installing dependencies..."
	.venv\Scripts\python.exe -m pip install --upgrade pip
	.venv\Scripts\python.exe -m pip install -r requirements.txt
	@echo "Setup complete! Activate with: .venv\Scripts\activate"

# Install/update dependencies only
install:
	@echo "Installing/updating dependencies..."
	.venv\Scripts\python.exe -m pip install --upgrade pip
	.venv\Scripts\python.exe -m pip install -r requirements.txt

# Clean up generated files
clean:
	@echo "Cleaning up..."
	if exist .venv rmdir /s /q .venv
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "Cleanup complete!"

# Check code quality
lint:
	@echo "Running code quality checks..."
	.venv\Scripts\python.exe -m ruff check .
	.venv\Scripts\python.exe -m black --check .

# Auto-format code
format:
	@echo "Formatting code..."
	.venv\Scripts\python.exe -m black .
	.venv\Scripts\python.exe -m ruff check --fix .

# Run tests
test:
	@echo "Running tests..."
	.venv\Scripts\python.exe -m pytest -q

# Start application
run:
	@echo "Starting Streamlit application..."
	.venv\Scripts\python.exe -m streamlit run app/app.py