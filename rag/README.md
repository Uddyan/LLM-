# Retrieval-Augmented Generation (RAG)

This directory contains the RAG pipeline for providing context-aware responses.

## Structure

- **retrieval/**: Context retrieval components
  - Vector database integration (Pinecone/Weaviate)
  - Hybrid search (keyword + vector)
  - Reranking models
  - Query processing

- **generation/**: Response generation
  - Prompt templates
  - Context building
  - Response synthesis
  - Citation management

## Benefits of RAG

- Always up-to-date information
- Reduces hallucinations
- More cost-effective than fine-tuning
- Easier to maintain

## Technologies

- Pinecone/Weaviate/Chroma (Vector databases)
- Sentence transformers
- Cross-encoder rerankers
- LangChain
