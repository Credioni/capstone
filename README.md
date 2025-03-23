# Project Schedule


| First Week    | Second Week   | Third Week   | Fourth Week |
| ------------- | ------------- |------------- |-------------|
| ✅Planning    | ✅Model + UI Skeletons | ✅Integration |  ✅Polishing   |
# Project 17: Multimedia Large Language Model Applications with Multimedia Embedding RAG


## Basic Idea
RAG models primarily focus on text-based retrieval, limiting their effectiveness in multimedia
applications. This project enhances RAG with multimodal retrieval by incorporating image, video, and
text embeddings. By integrating image and text embeddings in queries, the system enables faster and
more efficient retrieval, even on low-bandwidth networks. The use of compact embeddings ensures
quick access to relevant multimedia content, making AI-generated responses more contextually rich
and diverse

Create a scientific research assistant that retrieves related text, images, and videos from academic
papers.

## Requirements & Installation [Justfile](https://github.com/casey/just)
Make sure these are installed:\
1. [Node.js](https://nodejs.org/en)
2. [Python 3.12.X](https://www.python.org/downloads/)
3. Recommended [Justfile](https://github.com/casey/just) (or try install with Node.js with `npm install -g just-install`)


## Setup with `just`

Using `just` you can **install**, **build**, and retrive **faiss embeddings** using command:
```just
just setup-all
```

## Setup without `just`
**Install backend dependencies**
```bash
py -3.12 -m pip install -r requirements.txt
```
**Retrieve FAISS indices from HuggingFace**
```bash
py -3.12 .\retrieve-faiss.py
```
**Install node-packages**
```
cd frontend
pnpm install
```

## Test everything works
**Test that the FAISS retrieving is working using zero-shot**
```bash
just zero-shot
# cd backend/RAGembedder | py -3.12 .\zero_shot.py
```

**Start Frontend & Backend:**
```bash
just run-front
# cd frontend | pnpm run
```
```bash
just run-back
# cd backend | py -3.12 .\api.py
```
# Main files of the System
**FAISS** -contains all the handling and searching embeddings of indices:\
`multi_modal_embedder.py`

**LLM** -handles answer generation\
`arxiv_rag_system.py`

**Backend Handler** - Handles queue's request's from **frontend**\
`api_queue.py`

# Technologies and Tecniques used
## Frontend
1. `React.js` - with **Typescript**, **Rsbuild** with **Tailwindcss**.
2. `Material UI` - components of frontend.
3. [SVG -noise](https://css-tricks.com/grainy-gradients/)


## Backend
Tecniques and Matrial arts used
1. [HTTP caching]() -for fast request retrival.
2. [RAG-retrival]() caching.
3. [Multithreading]() & [Hash-Cache]() for even faster faiss-retrival.
