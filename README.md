# AI PDF Chatbot using Endee (RAG)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using the Endee vector database.
It allows users to upload PDFs (and other text documents) and ask questions based on document content.

## Features
- PDF upload
- Text chunking
- Embedding generation
- Vector storage in Endee
- Semantic search
- Answer generation from retrieved context

## Tech Stack
- Python
- Streamlit
- Google GenAI (`google-genai`) for embeddings + answering
- Endee Vector Database

## How it Works
1. PDF is converted into text
2. Text is split into chunks
3. Chunks are converted into embeddings
4. Stored in Endee
5. User query is embedded and matched using vector similarity (semantic search)
6. Retrieved chunks are used as context to generate the final answer

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```