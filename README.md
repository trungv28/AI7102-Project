# Legal Document Retrieval using Deep Learning methods

A sophisticated legal document retrieval system using Deep Learning methods to understand complex Vietnamese legal document relationships. The system implements two complementary approaches: knowledge distillation and GNN-based reranking.

## Project Overview

This project addresses legal document retrieval using the VLegalKMaps dataset (Vietnamese legal knowledge graphs). The system processes legal documents stored as RDF/Turtle graphs and retrieves relevant document chunks based on natural language queries.

### Key Features
- **Legal Document Processing**: Handles Vietnamese legal documents with complex hierarchical structures
- **Graph Neural Networks**: Uses Heterogeneous Graph Transformers (HGT) for legal document understanding
- **Knowledge Distillation**: Implements teacher-student training for improved embeddings

## System Architecture

### Approach 1: Knowledge Distillation Pipeline
A 5-round teacher-student distillation process:
- **Round 1**: Train embedder on gold data → train cross-encoder on gold labels
- **Rounds 2-3**: Teacher-guided training using cross-encoder generated labels
- **Rounds 4-5**: Self-improvement using embedder scores as pseudo-labels

### Approach 2: GNN-based Reranking (Primary Focus)
1. **Vector Search**: FAISS-based retrieval of candidate document chunks
2. **Graph Construction**: Builds heterogeneous legal graphs from retrieved candidates
3. **GNN Reranking**: Uses Heterogeneous Graph Transformers:
   - **DualReranker**: Combines local and global graph representations

## Project Structure

```
AI7102-Project/
├── src/                          # Main source code
│   ├── build_faiss.py           # FAISS vector database builder
│   ├── helper.py                # Utility functions and helpers
│   ├── synthesis/               # Data synthesis and dataset creation
│   │   ├── classifier.py       # Node classification for legal documents
│   │   ├── data_loader.py      # Knowledge graph data loading
│   │   ├── generator.py        # Question generation from legal graphs
│   │   └── prompt.py           # Prompt templates for LLM-based generation
│   ├── training/                # Training and model development
│   │   ├── base_distill.py      # Knowledge distillation pipeline
│   │   ├── dataset.py          # Dataset creation for GNN training
│   │   ├── loss.py             # Loss functions for reranking
│   │   ├── splits.py           # Train/validation/test splitting
│   │   ├── train.py            # Main training script
│   │   ├── train_loop.py       # Training loop implementation
│   │   └── train_utils.py      # Training utilities and evaluation
│   ├── models/                  # Neural network models
│   │   ├── hgt.py              # Heterogeneous Graph Transformer (HGT)
│   │   ├── pool.py             # Pooling mechanisms
│   │   ├── reranker.py         # GNN reranker implementation
│   │   └── reranker2.py        # Dual reranker variant
│   └── create_graph/           # Graph construction utilities
│       ├── create_graph.py     # Main graph builder
│       ├── graph_utils.py      # Graph construction utilities
│       ├── rdf_utils.py        # RDF/Turtle utilities
│       └── store.py            # Graph storage management
├── dataset/                     # Core dataset and processed data files
│   ├── dataset.jsonl           # Synthetic questions with ground truth URIs for training/evaluation
│   ├── splits.json             # Train/validation/test splits for reproducible experiments
│   ├── legal_graphs/           # Pre-computed heterogeneous graphs for GNN training (offline processing)
│   ├── vectordb_files/         # FAISS vector database and document metadata
│   │   ├── database.pkl      # All leaf nodes with text content and corresponding URIs
│   │   └── vectordb.bin      # FAISS index for fast similarity search of legal documents
│   ├── ttldataAJZ/             # Vietnamese legal documents - AJZ category (RDF/Turtle format)
│   ├── ttldataDEC/             # Vietnamese legal documents - DEC category (Decrees, RDF/Turtle format)
│   └── ttldataDOC/             # Vietnamese legal documents - DOC category (Documents, RDF/Turtle format)
├── weight/                      # Trained model weights
├── pyproject.toml               # Project configuration and dependencies
└── requirements.txt             # Full dependency list
```

## Quick Start

### Installation

Using `uv` (recommended):
```bash
# Clone the repository
git clone <repository-url>
cd AI7102-project

# Install dependencies
uv sync
```

Using `pip`:
```bash
pip install -r requirements.txt
```

### Data Setup

1. **Download Legal Dataset**: Download the data from https://aseados.ucd.ie/vlegal/download/. Place Vietnamese legal documents (.ttl files) in `dataset/` folder
2. **Generate Training Questions**:
   ```bash
   python -m src.synthesis.generator \
       --data-root dataset/ttldataAJZ \
       --outfile dataset/dataset.jsonl \
       --per-phase 10 \
       --model qwen2.5-7b-instruct
   ```
3. **Build Vector Database**:
   ```bash
   python -m src.build_faiss
   ```

### Training

#### Base Retriever (Traditional Embedding-based)
```bash
python -m src.training.train --task base_retr
```

#### GNN Reranking Approaches
```bash

# Dual GNN reranking (recommended)
python -m src.training.train --task gnn_dual
```

#### Knowledge Distillation
```bash

# For common models
python -m src.training.train --task base_teacher

# For bge-m3
python -m src.training.train --task bge_teacher
```

### Graph Construction for GNN Training
```bash
python -m src.create_graph.create_graph \
    --dataset-file dataset/dataset.jsonl \
    --max-samples 1000 \
    --max-cands 50 \
    --out-dir dataset/legal_graphs
```

## Key Components Explained

### Data Pipeline (`src/synthesis/`)

**`generator.py`**: Generates synthetic training questions from legal knowledge graphs in three phases:
- **Phase 1**: Content node questions from leaf nodes
- **Phase 2**: Structural questions from document articles
- **Phase 3**: Cross-document questions from connected documents

**`classifier.py`**: Classifies legal document nodes into:
- **Content Nodes**: Main leaf nodes containing legal text
- **Pointer Nodes**: Nodes indicating document relationships (replacement, amendment, citation)
- **Structural Nodes**: Non-leaf hierarchical nodes

### Graph Construction (`src/create_graph/`)

**`create_graph.py`**: Offline graph builder that:
1. Retrieves candidate nodes using FAISS vector search
2. Constructs local graphs for each candidate (full document tree + 1-hop documents)
3. Builds global graphs by combining all local graphs
4. Stores graphs with pre-computed BERT embeddings

**Graph Types**:
- **Local Graphs**: Complete document hierarchy + related documents for each candidate
- **Global Graph**: Union of all local graphs with inter-document connections

### Training Pipeline (`src/training/`)

**`train.py`**: Main training entry point supporting multiple approaches:
- `base_retr`: Traditional embedding-based retrieval
- `gnn_dual`: Dual reranking with local and global graphs
- `base_teacher`: Teacher-student knowledge distillation
- `bge_teacher`: Teacher-student knowledge distillation for BGE-M3

**`base_distill.py`**: 5-round knowledge distillation implementing iterative improvement through teacher-student training and self-supervision.


## Finetuned model weight

We provided with our best finetuned model GTE model here: https://huggingface.co/trung11/legal_finetuned_GTE
Download the weight and you can now use it for inference.