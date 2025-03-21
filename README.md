# Project Schedule


| First Week    | Second Week   | Third Week   | Fourth Week |
| ------------- | ------------- |------------- |-------------|
| ✅Planning    | ✅Model + UI Skeletons | ✅Integration |  🚧Polishing   |

# Usefull commands using `Justfile`

## Build faiss embeddings
```bash
just build-faiss
```


# Start frontend & backend
```bash
just run-front
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

# Technologies Used
## Frontend

## Backend
