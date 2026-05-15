# RAPTOR-Style Hierarchical Retrieval and Summarization Pipeline

## Overview

This project demonstrates a simplified implementation of a **RAPTOR-style** retrieval architecture using:

* **Sentence Transformers** for semantic embeddings
* **KMeans clustering** for grouping related documents
* **OpenAI GPT models** for cluster summarization
* **Vector similarity search** for query answering

The pipeline organizes technical documents into semantic clusters, generates high-level summaries for each cluster, and retrieves the most relevant summary for a user query.

This serves as a lightweight prototype of hierarchical retrieval systems used in modern Retrieval-Augmented Generation (RAG) architectures.

---

# Features

* Semantic document embeddings using transformer models
* Automatic clustering of related content
* LLM-generated hierarchical summaries
* Query-time semantic retrieval
* Modular pipeline design
* Easy to extend for large-scale RAG systems

---

# Architecture

```text
Documents
   │
   ▼
Sentence Embeddings
   │
   ▼
KMeans Clustering
   │
   ▼
Cluster Summaries (LLM)
   │
   ▼
RAPTOR Tree Layer
   │
   ▼
Semantic Query Retrieval
```

---

# Technologies Used

| Component            | Technology              |
| -------------------- | ----------------------- |
| Embeddings           | `sentence-transformers` |
| Clustering           | `scikit-learn`          |
| LLM Summarization    | `OpenAI GPT-4.1-mini`   |
| Numerical Operations | `NumPy`                 |

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/raptor-demo.git
cd raptor-demo
```

---

## 2. Install Dependencies

```bash
pip install numpy scikit-learn sentence-transformers openai
```

---

## 3. Set OpenAI API Key

### Linux / macOS

```bash
export OPENAI_API_KEY="your_api_key"
```

### Windows

```powershell
setx OPENAI_API_KEY "your_api_key"
```

---

# Code Walkthrough

## 1. Embedding Generation

```python
def embed_texts(texts):
    return embedding_model.encode(texts)
```

Uses the `all-MiniLM-L6-v2` transformer model to generate dense vector embeddings for semantic similarity tasks.

---

## 2. Document Clustering

```python
def cluster_texts_kmeans(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels
```

Groups semantically similar documents using KMeans clustering.

---

## 3. Cluster Summarization

```python
def summarize_cluster(texts):
```

Each cluster is summarized using an OpenAI GPT model into a high-level technical theme.

Example output:

```text
"Concepts related to scalable distributed infrastructure and cloud-native deployment systems."
```

---

## 4. RAPTOR Tree Construction

```python
def build_raptor_tree(texts):
```

Pipeline stages:

1. Generate embeddings
2. Cluster documents
3. Group texts by cluster
4. Summarize each cluster

Returns a list of cluster summaries representing the first hierarchical level of the RAPTOR tree.

---

## 5. Semantic Query Retrieval

```python
def query_raptor(query, summaries):
```

The query is embedded and compared against summary embeddings using dot-product similarity.

The most relevant summary is returned as the final answer.

---

# Example Documents

```python
documents = [
    "Microservices architecture improves scalability and deployment independence.",
    "Kubernetes manages container orchestration at scale.",
    "Distributed systems face consistency and partition tolerance challenges.",
    "Vector databases store embeddings for semantic search.",
    "RAG combines retrieval with generation for better answers.",
    "CI/CD pipelines automate deployment and reduce human errors.",
    "Load balancing distributes traffic across servers.",
    "Caching improves performance by reducing repeated computation."
]
```

---

# Example Query

```python
query = "How do large systems scale efficiently?"
```

### Example Output

```text
Concepts related to distributed systems scalability, orchestration, load balancing,
microservices, and infrastructure optimization.
```

---

# Running the Project

```bash
python main.py
```

---

# Sample Output

```text
Building RAPTOR tree...

--- Cluster Summaries ---
Infrastructure scalability and cloud-native deployment concepts.

Semantic retrieval and vector database technologies.

--- RAPTOR Answer ---
Infrastructure scalability and cloud-native deployment concepts.
```

---

# How RAPTOR Works

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is a retrieval architecture where:

1. Documents are recursively clustered
2. Each cluster is summarized
3. Summaries form higher-level representations
4. Queries traverse the hierarchy for efficient retrieval

This demo implements a simplified **single-level RAPTOR tree**.

---

# Possible Improvements

## Multi-Level Recursive Tree

Current implementation:

```text
Documents → Summaries
```

Future enhancement:

```text
Documents
   ↓
Cluster Summaries
   ↓
Meta-Cluster Summaries
   ↓
Global Knowledge Summary
```

---

## Better Similarity Search

Replace:

```python
np.dot(summary_embeddings, query_embedding)
```

With:

* Cosine similarity
* FAISS
* ChromaDB
* Pinecone
* Weaviate

---

## Dynamic Cluster Count

Instead of fixed:

```python
n_clusters=2
```

Use:

* Elbow method
* Silhouette score
* HDBSCAN

---

## Persistent Vector Storage

Integrate vector databases:

* FAISS
* Chroma
* Qdrant
* Milvus

---

# Real-World Applications

* Enterprise semantic search
* Retrieval-Augmented Generation (RAG)
* Knowledge management systems
* Technical documentation assistants
* AI research copilots
* Large-scale document understanding

---

# Project Structure

```text
raptor-demo/
│
├── main.py
├── README.md
└── requirements.txt
```

---

# requirements.txt

```text
numpy
scikit-learn
sentence-transformers
openai
```

---

# Limitations

* Single-level hierarchy only
* Small-scale demo dataset
* No persistent vector storage
* No recursive summarization
* Basic similarity search

---

# Future Enhancements

* Recursive RAPTOR hierarchy
* Streaming ingestion
* GPU acceleration
* Hybrid search (BM25 + vectors)
* Metadata-aware retrieval
* Interactive visualization
* Distributed embedding pipelines

---

# References

* RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
* Sentence Transformers
* OpenAI API
* scikit-learn KMeans
* Retrieval-Augmented Generation (RAG)

---

# License

MIT License

---

# Author

Venkata Sailesh Kumar Kilari
