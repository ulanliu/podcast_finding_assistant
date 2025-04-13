# Podcast "Commute For Me" 台灣通勤第一品牌 Content Finding Assistant

## Introduction

This project is a semantic search system designed to help users find relevant content from the Taiwanese podcast "Commute For Me" (台灣通勤第一品牌). It uses advanced natural language processing techniques to enable users to search for podcast episodes based on topics, key phrases, or concepts they remember, even if they don't know the exact episode number or title.

## Features

- **Two-stage Semantic Search**: Combines vector search with reranking for high-quality results
- **Multilingual Support**: Optimized for Traditional Chinese and English queries
- **Interactive Query Interface**: Simple command-line interface for searching the podcast archive
- **Performance Evaluation**: Built-in tools to measure search quality against ground truth data

## Architecture

The system consists of the following components:

1. **Vector Database (Qdrant)**: Stores embeddings of podcast episodes for fast retrieval
2. **Initial Retrieval Model**: Uses `intfloat/multilingual-e5-base` embeddings for first-stage retrieval
3. **Reranking Model**: Uses `BAAI/bge-reranker-base` for precise semantic matching
4. **Python Interface**: Handles queries and integrates the retrieval components

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Required Python packages (see requirements.txt)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   ```

2. Install Python dependencies:
   ```
   pip install -r requirement.txt
   ```

3. Start the vector database using Docker Compose:
   ```
   docker-compose up -d
   ```

4. Initialize the search system with podcast data:
   ```
   python semantic_search_rerank.py --mode init
   ```

### Usage

To search for podcast episodes:

```
python semantic_search_rerank.py --mode search
```

This will start an interactive console where you can enter search queries and see ranked results.

To evaluate search performance against ground truth data:

```
python semantic_search_rerank.py --mode evaluate
```

## Data Processing

The project includes notebooks for data processing:

- `ground_truth_data.ipynb`: Creates ground truth data using GPT-4o-mini
  - Generates query terms that should return specific episodes
  - Creates topic mappings for each episode

## System Components

### SemanticSearchRetriever Class

The core component that provides:

- Vector search with multilingual embedding model
- Cross-encoder reranking for better semantic matching
- Detailed search results with explanations
- Evaluation metrics (hit rate, MRR)

### Docker Services

- **Qdrant**: Vector database for storing and retrieving embeddings
- **Ollama**: Optional local model serving (for future development)

## Development and Evaluation

The system is evaluated using:
- Hit Rate (Recall@k): Measures if the correct episode appears in the top k results
- Mean Reciprocal Rank (MRR): Measures where in the result list the correct episode appears

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add license information here]

## Acknowledgments

- Thanks to the "Commute For Me" podcast team for creating the content
- Thanks to the open-source NLP community for the models and tools used