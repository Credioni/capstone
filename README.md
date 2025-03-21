# Project Schedule


| First Week    | Second Week   | Third Week   | Fourth Week |
| ------------- | ------------- |------------- |-------------|
| ✅Planning    | ✅Model + UI Skeletons | ✅Integration |  🚧Polishing   |

# Usefull commands using [Justfile](https://github.com/casey/just)
Scripts work in Windows enviroment and may work also in other platforms.
For script content see `justfile`-file.

**Build Faiss-embeddings** -with empty text/arXiv-embeddings
```bash
just build-faiss
```

**Start Frontend & Backend**.
```bash
just run-front
```
```bash
just run-back
```

# Project 17: Multimedia Large Language Model Applications with Multimedia Embedding RAG

## Basic Idea
RAG models primarily focus on text-based retrieval, limiting their effectiveness in multimedia
applications. This project enhances RAG with multimodal retrieval by incorporating image, video, and
text embeddings. By integrating image and text embeddings in queries, the system enables faster and
more efficient retrieval, even on low-bandwidth networks. The use of compact embeddings ensures
quick access to relevant multimedia content, making AI-generated responses more contextually rich
and diverse

## Requirements
Frontend
```
pip install faiss-cpu torch transformers sentence-transformers langchain langchain_huggingface pypdf pydantic pillow robyn colorlog
```

## Application
Create a scientific research assistant that retrieves related text, images, and videos from academic
papers.

# Technologies and Tecniques used
## Frontend
 - Main framework `React`, `Rsbuild` with `Tailwindcss`.

Tecniques used
1. [SVG -noise](https://css-tricks.com/grainy-gradients/) for more natural looking UI.
2. [Material UI]().
3. [Tailwindcss]().


## Backend
Tecniques and Matrial arts used
1. [HTTP caching]() -for fast request retrival.
2. [RAG-retrival]() caching.
3. [Multithreading]() & [Hash-Cache]() for even faster faiss-retrival.
