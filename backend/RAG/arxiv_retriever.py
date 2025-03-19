#pylint: disable=E0110,E0611,W1203,W1514
from typing import List
import os
import re
import json
import logging
import glob
import faiss
import torch
import numpy as np
from langchain.schema import BaseRetriever, Document
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from pypdf import PdfReader

# Set up logging
logger = logging.getLogger(__name__)

class ArXivRetriever(BaseRetriever, BaseModel):
    """
    A retriever class for arXiv papers that uses FAISS for similarity search.

    This retriever loads documents from a FAISS index, maps them to their
    original sources, and retrieves the most relevant documents for a query.
    """

    index_path: str
    mapping_path: str
    projection_path: str
    image_folder: str
    max_tokens: int = Field(default=3000)
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")

    # Non-serializable fields (excluded from Pydantic model)
    index: faiss.Index = Field(default=None, exclude=True)
    id_to_doc: dict = Field(default_factory=dict, exclude=True)
    projection: torch.nn.Module = Field(default=None, exclude=True)
    text_encoder: SentenceTransformer = Field(default=None, exclude=True)
    text_tokenizer: AutoTokenizer = Field(default=None, exclude=True)


    def __init__(self, **data):
        """Initialize the retriever with the given configuration."""
        super().__init__(**data)

        # Set the device
        torch.set_default_device(self.device)

        # Initialize components
        self._initialize_index()
        self._initialize_mapping()
        self._initialize_projection()
        self._initialize_models()

    def _initialize_index(self):
        """Initialize the FAISS index."""
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index from\n {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise

    def _initialize_mapping(self):
        """Load the document mapping from IDs to metadata."""
        try:
            with open(self.mapping_path) as f:
                self.id_to_doc = json.load(f)
            logger.info(f"Loaded document mapping from\n {self.mapping_path}")
        except Exception as e:
            logger.error(f"Failed to load document mapping: {e}")
            raise

    def _initialize_projection(self):
        """Initialize the projection layer for embeddings."""
        try:
            self.projection = torch.nn.Linear(384, 512)
            state_dict = torch.load(
                self.projection_path,
                weights_only=True,
                map_location=torch.device(self.device)
            )
            self.projection.load_state_dict({k.replace("linear.", ""): v for k, v in state_dict.items()})
            self.projection.eval()
            logger.info(f"Loaded projection layer from\n {self.projection_path}")
        except Exception as e:
            logger.error(f"Failed to load projection layer: {e}")
            raise

    def _initialize_models(self):
        """Initialize the text encoder and tokenizer models."""
        try:
            self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            self.text_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            logger.info("Loaded text encoder and tokenizer models")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query text into the vector space.

        Args:
            text: The query text to embed

        Returns:
            The normalized query embedding as a numpy array
        """
        try:
            query_embedding = self.text_encoder.encode(text)
            with torch.no_grad():
                query_embedding = self.projection(torch.tensor(query_embedding, device=self.device))

            query_embedding = query_embedding.cpu().numpy()
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            return query_embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def find_document_path(self, paper_id: str) -> str:
        """
        Find the document path for a given paper ID.

        Args:
            paper_id: The ID of the paper

        Returns:
            The path to the PDF file
        """
        try:
            pdf_files = glob.glob(f"data/pdfs/{paper_id}.pdf")
            if not pdf_files:
                raise FileNotFoundError(f"No PDF file found for paper ID {paper_id}")
            return pdf_files[0]
        except Exception as e: #pylint: disable=all
            logger.error(f"Error finding document path: {e}")
            return f"Error: {str(e)}"

    def extract_text_from_pdf(self, pdf_path: str, max_chars: int = 1000) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            max_chars: Maximum number of characters to extract

        Returns:
            The extracted text
        """
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file does not exist: {pdf_path}")
            return f"No PDF file available at {pdf_path}"

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PdfReader(file)
                text_parts = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())

                text = "\n".join(text_parts)
                # Clean up text by removing excess whitespace
                text = re.sub(r'\s+', ' ', text)
                return text[:max_chars]  # Truncate to max_chars
        except Exception as e: #pylint: disable=all
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return f"Error extracting text: {str(e)}"

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Get relevant documents for a query.

        Args:
            query: The query string
            **kwargs: Additional arguments:
                - k: Number of documents to retrieve (default: 10)

        Returns:
            A list of relevant documents
        """
        logger.warning(f"        {kwargs = }")

        k = kwargs.get("k", 3)
        logger.info(f"Retrieving {k} documents for query: {query}")

        try:
            query_embedding = self.embed_query(query)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)

            documents = []
            total_tokens = 0

            for idx, score in zip(indices[0], distances[0]):
                # Skip documents not in our mapping
                if str(idx) not in self.id_to_doc:
                    continue

                doc = self.id_to_doc[str(idx)]
                metadata = {
                    "type": doc.get("type", "unknown"),
                    "paper_id": doc.get("paper_id", "unknown"),
                    "score": float(score),
                }

                # Find the document path
                metadata["path"] = self.find_document_path(metadata["paper_id"])

                # Process content based on type
                if metadata["type"] == "image":
                    metadata["path"] = os.path.join(self.image_folder, os.path.basename(metadata["path"]))
                    content = f"Figure from {metadata['paper_id']}"
                else:
                    # Extract text from PDF
                    text = self.extract_text_from_pdf(metadata["path"])
                    content = text or f"No text available for {metadata['paper_id']}"

                    # Count tokens and truncate if necessary
                    tokens = self.text_tokenizer.encode(content, add_special_tokens=False)
                    if len(tokens) > 300:  # Truncate individual documents
                        tokens = tokens[:300]
                        content = self.text_tokenizer.decode(tokens)

                    # Stop if we've reached the maximum token count
                    if (total_tokens + len(tokens)) > self.max_tokens:
                        break

                    total_tokens += len(tokens)

                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            # Return an empty list in case of error
            return []

