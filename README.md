# RAG Chroma Agent

A Retrieval-Augmented Generation project using ChromaDB for document retrieval and answer generation.

## Overview
This repository demonstrates a simple RAG-based AI workflow where documents are stored in a vector database and later queried to produce enriched context-aware responses.

## Project Structure
- `User_input.py` - user interaction layer
- `system_setup.py` - project setup and configuration
- `mian.py` - main entry point
- `Database/` - database-related files
- `document/` - source documents
- `requirements.txt` - project dependencies
- `.gitignore` - ignored files

## Features
- ChromaDB integration
- Document-based retrieval
- Semantic search and context augmentation
- LLM-backed answer generation

## Getting Started
1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare the document store.
5. Run the project:

```bash
python mian.py
```

## Notes
This project is intended for learning and prototyping retrieval-augmented AI systems with vector search.
