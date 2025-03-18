#pylint: disable=E0110,E0611
from typing import List
import os
import re
import json
import faiss
import torch
from langchain.schema import BaseRetriever, Document
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from pypdf import PdfReader
import glob

torch.set_default_device("cuda")

# Custom Retriever Class
class ArXivRetriever(BaseRetriever, BaseModel):
    index_path: str
    mapping_path: str
    projection_path: str
    image_folder: str

    # Non-serializable fields (excluded from Pydantic model)
    index: faiss.Index = Field(default=None, exclude=True)
    id_to_doc: dict = Field(default_factory=dict, exclude=True)
    projection: torch.nn.Module = Field(default=None, exclude=True)
    text_encoder: SentenceTransformer = Field(default=None, exclude=True)
    text_tokenizer: AutoTokenizer = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)

        # Initialize FAISS index
        self.index = faiss.read_index(self.index_path)

        # Load id_to_doc mapping
        with open(self.mapping_path) as f:
            self.id_to_doc = json.load(f)

        # Load projection layer
        self.projection = torch.nn.Linear(384, 512)
        state_dict = torch.load(self.projection_path, weights_only=True)
        self.projection.load_state_dict({k.replace("linear.", ""): v for k, v in state_dict.items()})
        self.projection.eval()

        # Load text embedding model
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    def embed_query(self, text):
        """embed_query"""
        query_embedding = self.text_encoder.encode(text)
        with torch.no_grad():
            query_embedding = self.projection(torch.tensor(query_embedding))#.numpy()

        #faiss.normalize_L2(query_embedding.reshape(1, -1))
        query_embedding = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        return query_embedding

    def doc_path(self, id):
        """Find doct path in data with given id"""
        # print("current path", os.getcwd())
        pdf_file = glob.glob(f"data/pdfs/{id}.pdf")[0]
        return pdf_file

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """embed_query"""
        k = kwargs.get("k", 10)
        query_embedding = self.embed_query(query)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)


        documents = []
        total_tokens = 0
        max_tokens = 3000

        documents = []
        for idx, score in zip(indices[0], distances[0]):

            # Filter only docs can pass
            if str(idx) not in self.id_to_doc:
                continue

            doc = self.id_to_doc[str(idx)]
            metadata = {
                "type": doc.get("type", "unknown"),
                "paper_id": doc.get("paper_id", "unknown"),
                # "path": doc.get("path", ""),
                "score": float(score),
            }
            metadata["path"] = self.doc_path(metadata["paper_id"])

            # Fix PDF paths
            # if metadata["type"] == "text":
            #     original_path = metadata["path"]
            #     # Corrected path replacement logic
            #     if "/kaggle/working/pdfs" in original_path:
            #         metadata["path"] = original_path.replace(
            #             "/kaggle/working/pdfs",
            #             "/kaggle/input/outputfiles/kaggle/working/pdfs"
            #         )

            # Process content
            if metadata["type"] == "image":
                metadata["path"] = os.path.join(self.image_folder, os.path.basename(metadata["path"]))
                content = f"Figure from {metadata['paper_id']}"
            else:
                text = ""
                if metadata["path"] and metadata["path"].endswith(".pdf"):
                    try:
                        if os.path.exists(metadata["path"]):
                            with open(metadata["path"], "rb") as file:
                                pdf_reader = PdfReader(file)
                                text_parts = []
                                for page in pdf_reader.pages:
                                    page_text = page.extract_text()
                                    if page_text and page_text.strip():
                                        text_parts.append(page_text.strip())
                                text = "\n".join(text_parts)
                                text = re.sub(r'\s+', ' ', text)[:1000]  # Truncate to 1000 characters
                    except Exception as e:
                        text = f"Error extracting text: {str(e)}"

                content = text or f"No text available for {metadata['paper_id']}"
                tokens = self.text_tokenizer.encode(content, add_special_tokens=False)
                if len(tokens) > 300:  # Truncate individual documents
                    tokens = tokens[:300]
                    content = self.text_tokenizer.decode(tokens)
                if (total_tokens + len(tokens)) > max_tokens:
                    break
                total_tokens += len(tokens)
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

        return documents