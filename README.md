# RAG Pipeline for Design System Documentation

A Retrieval-Augmented Generation (RAG) pipeline built with Haystack and Anthropic Claude to enhance search and discovery within design system documentation.

## Features

- Semantic search across design system documentation
- Component-aware chunking and indexing
- Context-aware responses using Claude 3.5 Sonnet
- Support for multiple documentation formats (Markdown, MDX, HTML, JSON)
- Docker-based deployment for easy setup

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Anthropic API key

### Setup

1. Clone the repository and navigate to the project directory

2. Copy the environment template and add your API keys:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

3. Start the services with Docker Compose:
```bash
docker-compose up -d
```

4. The API will be available at `http://localhost:8000`

### Local Development (without Docker)

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Project Structure

See `rag-plan.md` for detailed architecture and implementation plan.

## Development

- Run tests: `pytest tests/`
- Format code: `black src/`
- Lint: `ruff check src/`

## License

[Your License Here]
