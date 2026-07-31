# RAG Assistant for *Euclid's Book on Divisions of Figures*

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonDinkov/AI_integrationsForDev/blob/main/regular_exam/Regular_exam.ipynb)

This project implements a **Retrieval-Augmented Generation (RAG)** assistant for the PDF edition of *Euclid's Book on Divisions of Figures*. The assistant answers questions grounded in the indexed document and can present the result as **text, an educational image, or speech**.

The key idea is simple: **every response is generated from information retrieved from the PDF, not from the model's general knowledge.**

---

# System Architecture

```text
                     PDF Document
                          │
                          ▼
                Text Extraction (pypdf)
                          │
                          ▼
            Overlapping Text Chunking
                          │
                          ▼
          OpenAI Embedding Generation
                          │
                          ▼
                     ChromaDB
                          │
                          ▼
                 Semantic Retrieval
                          │
                          ▼
                 Grounded LLM Answer
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
        Text          Gemini Image    ElevenLabs Audio
```

## Document Indexing

The PDF is processed page by page using **pypdf**. Each page is split into overlapping chunks (`CHUNK_SIZE = 1400`, `CHUNK_OVERLAP = 250`) before being converted into embeddings with **OpenAI `text-embedding-3-small`** and stored in **ChromaDB**.

## Semantic Retrieval

When a user asks a question, the same embedding model converts the query into a vector. ChromaDB retrieves the six most relevant chunks, allowing semantic search instead of simple keyword matching.

---

# Why Does the Model Have Only One Tool?

The language model is intentionally given access to **only one tool**:

```python
retrieve_information()
```

The model **cannot** call image generation or audio generation directly.

Instead, the architecture separates **retrieval** from **presentation**.

```text
User Question
      │
      ▼
Request Parsing
      │
      ▼
retrieve_information()   ← the only LLM tool
      │
      ▼
Grounded Answer
      │
      ▼
Python Application
      │
 ┌────┼─────────────┐
 ▼    ▼             ▼
Text Image        Audio
```

## Why is one tool enough?

Regardless of whether the user requests text, an image, or audio, the system always needs the same factual information from the document.

Therefore, retrieval is performed **once**.

```text
Retrieve once
      │
      ▼
Ground once
      │
      ▼
Present as
├── Text
├── Image
└── Audio
```

The grounded answer is then reused:

- displayed directly as text;
- sent to **Gemini** to generate an educational illustration;
- sent to **ElevenLabs** to generate speech.

This keeps every output consistent because all modalities originate from the same verified document context.

---

# Separation of Responsibilities

### Language Model

- Understands the user's request.
- Resolves follow-up questions.
- Calls `retrieve_information()`.
- Generates a grounded answer using only the retrieved context.

### Python Application

- Controls the workflow.
- Selects the requested output format.
- Calls Gemini for image generation.
- Calls ElevenLabs for speech synthesis.
- Displays the final result.

This architecture keeps the retrieval pipeline deterministic while allowing multiple output formats without exposing unnecessary tools to the LLM.

---

# Forced Retrieval

The notebook explicitly forces retrieval before answer generation:

```python
tool_choice={
    "type": "function",
    "name": "retrieve_information"
}
```

This prevents the model from answering directly from pretrained knowledge and ensures that responses are grounded in the retrieved PDF content.

---

# Main Entry Point

```python
ask_ai(question)
```

Supported output formats:

- Text
- Image
- Audio

---

# Technologies

| Component | Technology |
|-----------|------------|
| PDF Processing | pypdf |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Database | ChromaDB |
| Retrieval & Answer Generation | OpenAI |
| Image Generation | Google Gemini |
| Speech Synthesis | ElevenLabs |
| Structured Output | Pydantic |

---

# Running the Notebook

1. Open `Regular_exam.ipynb` in Google Colab.
2. Add the required API keys.
3. Run the notebook.
4. Upload the PDF if prompted.
5. Wait for the ChromaDB index.
6. Start asking questions with `ask_ai(...)`.

---

# Key Design Decision

The project intentionally exposes **only one retrieval tool** to the language model.

The LLM is responsible only for retrieving factual information from the indexed document. The Python application is responsible for presenting that information as **text, image, or speech**.

This guarantees that every response—regardless of output format—is grounded in the same retrieved document context.
