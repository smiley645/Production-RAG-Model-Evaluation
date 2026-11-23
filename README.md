# Production RAG Model Evaluation

A production-oriented Retrieval-Augmented Generation (RAG) system built using four AI research papers. This project explores, implements, compares, and evaluates multiple embedding models, performs threshold-based semantic retrieval, and generates context-grounded answers using a Groq-hosted Large Language Model (LLM).

The primary objective is to build a reliable RAG pipeline that retrieves relevant research-paper content, generates accurate answers using retrieved context, and evaluates retrieval and generation quality through quantitative evaluation metrics.

## Project Overview

This project implements an end-to-end RAG workflow using four research documents. Three embedding models are evaluated independently to identify which performs best for semantic retrieval.

The system uses cosine similarity-based retrieval with individually calibrated thresholds for each embedding model. Retrieved text chunks are passed to a Groq-hosted LLM, which generates answers using the supplied research context and provides source and page citations.

The project also incorporates retrieval evaluation metrics and RAGAS-based Faithfulness evaluation.

## Research Documents

The knowledge base consists of four PDF research documents:

1. **Attention Is All You Need** – Transformer architecture and self-attention mechanisms.
2. **CNN Introduction** – Fundamentals and components of convolutional neural networks.
3. **Deep Residual Learning for Image Recognition (ResNet)** – Residual connections and deep neural network architectures.
4. **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (Vision Transformer)** – Vision Transformers and their application to image recognition.

Each PDF is processed independently, and source filename and page number metadata are retained throughout the retrieval pipeline.

## Technology Stack

| Component | Technology |
|---|---|
| Development Environment | Google Colab (Free Tier) |
| Programming Language | Python |
| PDF Processing | PyMuPDF |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embedding Framework | Sentence Transformers |
| Vector Database | ChromaDB |
| Vector Search | Cosine Similarity |
| LLM Integration | LangChain |
| LLM Provider | Groq |
| LLM Model | openai/gpt-oss-20b |
| Evaluation Framework | RAGAS |
| Data Processing | NumPy, Pandas |
| Model Execution | CPU / Optional GPU |

## Embedding Models

Three different embedding models are implemented and compared.

| Model | Hugging Face Model |
|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` |
| BGE | `BAAI/bge-small-en-v1.5` |
| E5 | `intfloat/e5-small-v2` |

Each model generates its own document and query embeddings.

**E5-specific configuration:** The E5 model uses the `passage:` prefix for document embeddings and the `query:` prefix for query embeddings, following its recommended input format.

All embeddings are normalized before similarity comparison.

## RAG Pipeline Architecture

The system follows this workflow:

1. **Document Ingestion:** Load the four research PDFs and extract page-wise text using PyMuPDF.
2. **Text Chunking:** Split extracted text into manageable chunks using RecursiveCharacterTextSplitter.
3. **Embedding Generation:** Generate embeddings independently using MiniLM, BGE, and E5.
4. **Vector Storage:** Store document chunks, embeddings, identifiers, and source metadata in separate ChromaDB collections.
5. **Similarity Search:** Calculate cosine similarity between the question embedding and document embeddings.
6. **Threshold-Based Retrieval:** Apply a separately calibrated similarity threshold for each embedding model.
7. **Context Preparation:** Combine accepted chunks along with their source filenames and page numbers.
8. **LLM Answer Generation:** Pass the retrieved context and user question to the Groq-hosted LLM.
9. **Source Attribution:** Generate answers with instructions to cite the relevant research document and page number.
10. **Evaluation:** Evaluate retrieval performance and answer faithfulness.

## Document Chunking Configuration

The following configuration is used:

| Parameter | Value |
|---|---:|
| Chunk Size | 1,000 characters |
| Chunk Overlap | 150 characters |
| Splitting Strategy | Recursive Character Splitting |
| Number of PDF Documents | 4 |
| Total Chunks Created | 243 |

Each chunk retains its source document and original page number as metadata.

## Vector Database

ChromaDB is used for vector storage and similarity retrieval.

Three independent collections are created:

- `minilm_collection`
- `bge_collection`
- `e5_collection`

Each collection contains the document identifiers, original chunk text, embedding vectors, and metadata.

Cosine distance is configured using:

`{"hnsw:space": "cosine"}`

This allows each embedding model to be evaluated independently.

## Similarity Threshold Calibration

Instead of using arbitrary similarity thresholds, a separate threshold is calibrated for each embedding model using the evaluation dataset.

The calibration procedure:

1. Generate query embeddings for evaluation questions.
2. Calculate cosine similarity scores against the document embeddings.
3. Label chunks according to their expected relevant research-paper source.
4. Evaluate different similarity cutoffs using precision and recall.
5. Calculate the F1-score for candidate thresholds.
6. Select the threshold that maximizes F1-score for each embedding model.

### Calibrated Thresholds

| Embedding Model | Similarity Threshold |
|---|---:|
| MiniLM | 0.3438 |
| BGE | 0.6609 |
| E5 | 0.8226 |

These thresholds are model-specific and should be recalibrated when the document collection or evaluation dataset changes.

The retrieval system applies the calibrated threshold first and then ranks accepted chunks by similarity. Top-K acts as the maximum number of chunks returned.

## LLM Integration and Answer Generation

The project integrates a Groq-hosted LLM using LangChain.

**LLM:** `openai/gpt-oss-20b`

The system uses:

- `ChatGroq`
- `ChatPromptTemplate`
- `StrOutputParser`

The RAG prompt includes strict grounding instructions:

- Answers must use only the retrieved research-paper context.
- External knowledge must not be introduced.
- Unsupported facts and assumptions must not be invented.
- Source filename and page number citations must be provided.
- When sufficient information is unavailable, the model must respond:

`I don't know based on the provided research documents.`

The retrieved context is passed directly to the LLM along with the user's question.

This helps reduce unsupported answers and provides traceability to the original research documents.

## Evaluation Framework

The system evaluates both retrieval quality and answer quality.

### 1. Retrieval Evaluation

The following metrics are implemented:

| Metric | Purpose |
|---|---|
| Hit Rate@5 | Measures whether at least one relevant source is retrieved within the top five results. |
| Precision@5 | Measures the proportion of retrieved positions belonging to the expected source document. |
| Recall@5 | Measures the proportion of chunks from the expected source document retrieved. |
| Mean Reciprocal Rank (MRR) | Measures how highly the first relevant result is ranked. |

The retrieval evaluation uses a manually prepared question dataset covering all four research papers.

### 2. RAGAS Evaluation

RAGAS is used to evaluate the quality of generated answers against retrieved context.

**Faithfulness:** Evaluates whether the factual claims in the generated answer are supported by the retrieved research-paper context.

Additional RAGAS evaluation components include Answer Relevancy and Context Precision without reference answers.

### 3. Evaluation Results

The final CPU-based retrieval evaluation produced the following results:

| Model | Hit Rate@5 | Precision@5 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| MiniLM | 1.00 | 0.78 | 0.0716 | 0.95 |
| BGE | 1.00 | 0.80 | 0.0686 | 0.95 |
| E5 | **1.00** | **0.86** | **0.0791** | **1.00** |

**Key Findings**

- All three embedding models achieved a Hit Rate@5 of 1.00 on the final five-question evaluation.
- E5 achieved the highest Precision@5.
- E5 achieved the highest Recall@5 among the three evaluated models.
- E5 achieved an MRR of 1.00, indicating that the first relevant source appeared at rank one for every evaluation question.
- E5 demonstrated the strongest overall retrieval performance in this evaluation.

**Evaluation Note:** Recall@5 is calculated at the source-level chunk coverage level, using the total number of chunks belonging to the expected source document as the denominator. These results are based on a small evaluation dataset and should not be interpreted as a universal benchmark.

RAGAS Faithfulness evaluation is also incorporated into the project to assess the factual consistency of generated answers with retrieved context.

## Project Configuration and Execution

The project was developed and executed in Google Colab.

### Main Libraries

```bash
pip install sentence-transformers
pip install pymupdf
pip install langchain
pip install langchain-community
pip install langchain-chroma
pip install chromadb
pip install langchain-groq
pip install ragas
pip install pandas numpy
```

The Groq API key is configured using Google Colab Secrets.

Do not hardcode API keys or upload private credentials to GitHub.

### Execution Workflow

1. Install the required dependencies.
2. Load the four research PDFs.
3. Extract text and generate chunks.
4. Load the MiniLM, BGE, and E5 embedding models.
5. Generate and store document embeddings.
6. Create the individual vector collections.
7. Prepare the retrieval evaluation dataset.
8. Calibrate model-specific similarity thresholds.
9. Evaluate retrieval performance.
10. Configure the Groq LLM and RAG prompt.
11. Run question-answer generation using retrieved context.
12. Perform RAGAS-based answer evaluation.

The retrieval and embedding evaluation can run on CPU, making the implementation suitable for Google Colab Free Tier environments. GPU acceleration is optional.

## Project Backup and Artifacts

A project backup archive is generated as:

`RAG_Project_Final.zip`

The backup includes:

- Original research-paper PDFs.
- Extracted document chunks and metadata.
- MiniLM embedding matrix.
- BGE embedding matrix.
- E5 embedding matrix.
- Model configurations.
- Calibrated similarity thresholds.
- Retrieval evaluation results.
- Evaluation questions.
- Final question-answer test records.
- Project completion summary.

The backup can be downloaded from Google Colab and retained locally.

Note that the Colab `/content` directory is temporary. Download the archive before the runtime is deleted.

## Key Features

- Multi-model embedding comparison.
- Independent vector collections for each embedding model.
- Cosine similarity-based semantic retrieval.
- Data-driven similarity threshold calibration.
- Threshold-based top-K retrieval.
- Source and page-level metadata tracking.
- Context-grounded LLM answer generation.
- Source attribution in generated answers.
- Retrieval performance evaluation.
- RAGAS-based Faithfulness evaluation.
- CPU-compatible execution.
- Exportable project artifacts.

## Future Improvements

- Expand the research-paper collection.
- Increase the size and diversity of the evaluation dataset.
- Perform independent threshold validation.
- Improve passage-level relevance annotations.
- Extend RAGAS evaluation across a larger answer dataset.
- Deploy the retrieval component as a reusable API.
- Integrate the RAG retrieval system as a tool within a multi-agent LLM architecture.
- Implement monitoring and logging for production deployment.

## Conclusion

Production RAG Model Evaluation demonstrates the implementation and evaluation of a complete retrieval-augmented generation workflow using scientific research documents.

By comparing MiniLM, BGE, and E5, calibrating similarity thresholds independently, evaluating retrieval performance, and integrating a context-aware Groq LLM, the project establishes a reusable research-paper retrieval and question-answering system.

The evaluation results identify E5 as the strongest-performing embedding model among the three models tested on the prepared evaluation dataset.

The resulting RAG component can also serve as a reusable knowledge-retrieval tool for larger LLM and multi-agent applications.

---

**Project Name:** Production RAG Model Evaluation  
**Environment:** Google Colab  
**Language:** Python  
**LLM Provider:** Groq  
**Embedding Models:** MiniLM, BGE, E5  
**Vector Database:** ChromaDB  
**Evaluation Framework:** RAGAS
