# Multimodal RAG AI Assistant

A **multimodal AI assistant** based on **Retrieval-Augmented Generation (RAG)** built with **Streamlit, LangChain, ChromaDB, and OpenAI**.

The application allows users to upload **PDF documents and images**, extract their content using **OCR**, and ask questions through an interactive chat interface.

The AI retrieves relevant information from the uploaded documents and generates contextual answers.

---

## Screenshot

<img width="1911" height="1040" alt="image" src="https://github.com/user-attachments/assets/16243a9a-e2c3-4b40-9e3f-8a7f2c870102" />

## Features

- Upload multiple **PDF documents**
- Upload **images**
- Extract text from PDFs using **PyPDF2**
- Extract text from images using **OCR (pytesseract)**
- Automatic **text chunking**
- Generate **OpenAI embeddings**
- Store embeddings in **Chroma vector database**
- Retrieve the most relevant document chunks
- Generate answers with **OpenAI LLM**
- Interactive **chat interface**
- Persistent **chat history**
- Input bar fixed at the bottom (chat-style interface)

---

## Architecture

```text
User
  ↓
Upload PDF / Images
  ↓
Text Extraction
  ├── PDF → PyPDF2
  └── Images → OCR with pytesseract
  ↓
Text Chunking
  ↓
Embeddings Generation (OpenAI)
  ↓
Chroma Vector Database
  ↓
Retriever
  ↓
LLM (OpenAI)
  ↓
Generated Answer in Chat Interface
```

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- PyPDF2
- pytesseract
- Pillow

---

## Project Structure

```text
multimodal-rag-ai-assistant
│
├── rag.py
├── ragmultiple.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── screenshots
```
