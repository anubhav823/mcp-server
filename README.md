# 🧠 MCP Agent with Multi-Skill LLM + RAG

## 🚀 Overview

This project implements a **Multi-Component Prompting (MCP) Agent** powered by the **Gemini API**, designed to handle multiple skills through modular prompt orchestration. The system integrates **Retrieval-Augmented Generation (RAG)** to provide context-aware, knowledge-grounded responses.

---

## 🏗️ Architecture

The system is composed of the following core components:

* **LLM Layer**: Gemini API for reasoning and response generation
* **RAG Pipeline**: Enhances responses with external knowledge
* **Indexing Framework**: LlamaIndex for data ingestion and query handling
* **Vector Database**: ChromaDB for semantic search and embeddings storage
* **MCP Orchestrator**: Routes queries to appropriate skills and prompts

---

## ⚙️ Features

### 🔹 Multi-Skill Agent

* Supports multiple capabilities (Q&A, summarization, domain-specific tasks)
* Modular prompt design for easy extension

### 🔹 Retrieval-Augmented Generation (RAG)

* Fetches relevant context from vector DB before LLM generation
* Improves factual accuracy and reduces hallucinations

### 🔹 Semantic Search with ChromaDB

* Stores embeddings for fast similarity search
* Enables context-aware retrieval

### 🔹 LlamaIndex Integration

* Efficient document ingestion and indexing
* Flexible query pipeline construction

---

## 🧩 System Flow

1. User query is received by MCP agent
2. Query is routed to the appropriate skill/prompt
3. Relevant documents are retrieved via ChromaDB
4. LlamaIndex constructs context for the query
5. Gemini LLM generates the final response using RAG

---

## 🛠️ Tech Stack

* **LLM**: Gemini API
* **Framework**: LlamaIndex
* **Vector DB**: ChromaDB
* **Language**: (Add your language, e.g., Python / Java)
* **Architecture**: Modular MCP + RAG

---

## 📈 Key Learnings

* Designing scalable LLM-based systems
* Implementing RAG for production use-cases
* Prompt engineering for multi-skill orchestration
* Efficient vector search and indexing strategies

---

## 🔮 Future Improvements

* Add tool usage / function calling support
* Introduce memory for conversational context
* Improve ranking and retrieval strategies
* Deploy as a scalable API service

---

## 📌 Getting Started

```bash
# Clone the repository
git clone <your-repo-link>

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```
