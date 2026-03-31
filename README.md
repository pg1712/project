
## Project Overview

This project implements a simplified version of **FLARE (Forward-Looking Active Retrieval Augmented Generation)** to reduce hallucination in Large Language Models (LLMs).

Traditional LLMs often generate incorrect information. While Retrieval-Augmented Generation (RAG) improves factual accuracy, it retrieves information only once.  

FLARE improves this by **dynamically triggering retrieval based on model confidence**.

##  Key Idea

- Use **token-level probability** as a confidence signal  
- If the model is uncertain → trigger retrieval  
- Use predicted sentence as a **forward-looking query**  
- Regenerate answer using new knowledge  

## Architecture

User Query
↓
BM25 Retriever
↓
Context + Query → LLM
↓
Generate Sentence
↓
Check Token Confidence
↓
[Low Confidence?]
├── Yes → Retrieve Again → Regenerate
└── No → Accept Sentence

##  Tech Stack

- Python  
- Hugging Face Transformers  
- BM25 (rank_bm25)  
- PyTorch  

## Dataset

- Custom dataset with **28 factual documents**
- Covers:
  - Geography  
  - Programming  
  - General knowledge  
- Stored in `data.txt`

---

## Methods Implemented

### 1️ No Retrieval
- Direct LLM generation  
- High hallucination  

### 2️ RAG (Single Retrieval)
- Retrieve once before generation  
- Improved factual accuracy  

### 3️ FLARE (Proposed)
- Dynamic retrieval based on confidence  
- Iterative generation  

##  Results

| Method | Exact Match (EM) |
|--------|-----------------|
| No Retrieval | 0.17 |
| RAG | 1.00 |
| FLARE | 1.00 |

##  Sample Output

###  No Retrieval

Question: Who is Joe Biden?

Output:
Joe Biden studied business at Fordham University...

###  RAG

Context: Joe Biden is the 46th President of the United States...

Output:
Joe Biden is the 46th President of the United States.
He studied at the University of Delaware and Syracuse University.

###  FLARE (Dynamic Retrieval)

[FLARE Retrieval Log]

Step 1: RETRIEVAL TRIGGERED (min_prob=0.104)
masked query: Joe Biden studied at
retrieved docs: ...

Step 2: accepted (min_prob=0.93)
final answer:
Joe Biden studied at the University of Delaware...

##  Key Insights

- LLMs alone may hallucinate  
- RAG significantly improves factual accuracy  
- FLARE enables **dynamic retrieval based on uncertainty**  
- Token probability is an effective confidence signal  

##  How to Run

1️Create Virtual Environment
python -m venv venv
source venv/bin/activate
2️ Install Dependencies
pip install transformers torch rank_bm25 nltk
3️ Run
python main.py
