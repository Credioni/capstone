
# Project 17: Multimedia Large Language Model Applications with Multimedia Embedding RAG

## Project overview
RAG models primarily focus on text-based retrieval, limiting their effectiveness in multimedia applications. This project enhances RAG with multimodal retrieval by incorporating image, video, and text embeddings. By integrating image and text embeddings in queries, the system enables faster and more efficient retrieval. With even on low-bandwidth networks. The use of **compact embeddings** ensures quick access to **relevant multimedia content**, making AI-generated responses more contextually rich and diverse.

**Project:**
A scientific research assistant that retrieves -based on query- relevant text, images, audio and videos from academic
sources, example from ArXiv.

# Technologies used
### Frontend
1. `React.js` - with **Typescript**, **Rsbuild** with **Tailwindcss**.
2. `Material UI` - components of frontend.
3. `Tailwindcss` - uses tailwindcss with traditional css.
3. Also with custom components.

### Backend
1. [LangChain](https://www.langchain.com) - Creating LLM and multi-media pipeline.
2. [Faiss](https://github.com/facebookresearch/faiss) - for dense vector-storage embeddings (used in the project).

[Logging]() - Backend implements error handling and terminal logging, with traceback. \
[HTTP caching]() - Multimedia requests are **hash-cached**.\
[Retrival-Hash-Caching]() - Hash caching queries and corresponding information for faster retrival.\
[Multithreaded]() -for `Faiss` retrivals and `LangChain` -interactions.

### Models used
| Modality  | Model                            | Dimensionality |
| --------- | ----------------------------------------- | -------------- |
| Textual   | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)    | 384            |
| Audio     | [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en)                   | 384            |
| Image     | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)              | 512            |
| Video     | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)              | 512            |
| Answer Generation | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | |

**Textual-Image modalities** are cross-searched using multi-modal `CLIP` -model with projection layer.
### About - FAISS embeddings
 - **Textual** -embeddings contains around [2.7million arXiv research paper](https://www.kaggle.com/datasets/Cornell-University/arxiv) -collected 03/2025, indexed using paper's title & abstract -collected.
 - **Image & Sound** -embeddings contains indexed debug data.
 - **Video** -embeddings contains context from **3Blue1Brown** youtube channel, indexed using (title and transcription data) + (average index of **4fps** frames).


# Installation instructions [Justfile](https://github.com/casey/just)
Make sure these are installed in the system, if not already:
1. [Node.js](https://nodejs.org/en)
2. [Python 3.12.X](https://www.python.org/downloads/)
3. Recommended [Justfile](https://github.com/casey/just) (fast-install with Node.js `npm install -g just-install`)


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

# Usage guide
## Test zero-shot
**Test that the FAISS retrieving is working using zero-shot**
```bash
just zero-shot
# cd backend/rag | py -3.12 .\zero_shot.py
```

## Running
**Start Frontend & Backend in separate terminals**
```bash
just run-back
# cd backend | py -3.12 .\api.py
```
```bash
just run-front
# cd frontend | pnpm run
```

Starting the frontend, open's up a browser to interact with the system!

<!-- # Main files of the System
**FAISS** -contains all the handling and searching embeddings of indices:\
`multi_modal_embedder.py`

**LLM** -handles answer generation\
`arxiv_rag_system.py`

**Backend Handler** - Handles queue's request's from **frontend**\
`api_queue.py` -->



