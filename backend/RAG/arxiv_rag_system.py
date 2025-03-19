#pylint: disable=E0110,E0611,W1203
import os
import logging
import glob
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .arxiv_retriever import ArXivRetriever

# Set up logging
logger = logging.getLogger(__name__)

class ArXivRAGSystem:
    """
    A RAG (Retrieval-Augmented Generation) system for arXiv papers.

    This system combines a retriever for fetching relevant documents and a
    language model for generating answers based on those documents.
    """

    def __init__(self, config):
        """
        Initialize the RAG system with the given configuration.

        Args:
            config: A dictionary containing configuration parameters:
                - faiss_index_path: Path to the FAISS index file
                - mapping_path: Path to the document mapping file
                - projection_path: Path to the projection model file
                - image_folder: Path to the folder containing images
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.retriever = self._initialize_retriever()
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()

    def _initialize_retriever(self):
        """Initialize the document retriever."""
        try:
            retriever = ArXivRetriever(
                index_path=self.config['faiss_index_path'],
                mapping_path=self.config['mapping_path'],
                projection_path=self.config['projection_path'],
                image_folder=self.config['image_folder'],
                device=self.device
            )
            logger.info("Initialized ArXiv retriever")
            return retriever
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

    def _initialize_llm(self):
        """Initialize the language model."""
        # Use phi-2 model for generation (smaller memory footprint)
        # self.model_name = "microsoft/phi-2"
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # self.model_name = "gpt2"

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )

            text_generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )

            llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info(f"Initialized language model: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise

    def _create_qa_chain(self):
        """Create the question-answering chain."""
        try:
            prompt_template = """Generate a detailed answer to the question below using the provided context.
            Include references to figures and papers where applicable.

            Context:\n\n{context}

            Question: {question}

            Structured Answer:"""

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=prompt_template,
                        input_variables=['context', 'question']
                    ),
                    "document_separator": "\n\n---\n\n"
                },
                return_source_documents=True
            )

            logger.info("Created question-answering chain")
            return qa_chain
        except Exception as e:
            logger.error(f"Failed to create QA chain: {e}")
            raise

    def find_document_path(self, paper_id):
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
                return None
            return pdf_files[0]
        except Exception as e:
            logger.error(f"Error finding document path: {e}")
            return None

    def clean_answer(self, text):
        """
        Clean the generated answer by removing duplicate sentences.

        Args:
            text: The raw generated text

        Returns:
            The cleaned text with duplicate sentences removed
        """
        # Split into sentences
        sentences = text.split('. ')
        seen = set()
        unique = []

        for sent in sentences:
            # Use first 50 chars as a key to detect duplicates
            key = sent[:50].lower() if len(sent) >= 50 else sent.lower()
            if key not in seen:
                seen.add(key)
                unique.append(sent)

        return '. '.join(unique)

    def query(self, question, k=5, score_threshold=0.6):
        """
        Query the RAG system with a question.

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            score_threshold: Threshold for filtering documents by score

        Returns:
            A dictionary containing:
                - answer: The generated answer
                - sources: The source documents used for the answer
                - images: Any images referenced in the sources
        """
        logger.info(f"Processing query: {question}")

        try:
            # Query the QA chain
            results = self.qa_chain.invoke(input=question, k=k)

            # Log information about retrieved documents
            logger.info(f"Retrieved {len(results['source_documents'])} documents")
            for doc in results['source_documents']:
                logger.debug(f"Document score: {doc.metadata['score']}, "
                            f"Paper ID: {doc.metadata['paper_id']}")

            # Process sources (lower distance = better match)
            sources = []
            for doc in results['source_documents']:
                if doc.metadata['score'] < score_threshold:  # Keep good matches
                    sources.append({
                        "type": doc.metadata['type'],
                        "paper_id": doc.metadata['paper_id'],
                        "score": doc.metadata['score'],
                        "content": doc.page_content,
                        "path": self.find_document_path(doc.metadata['paper_id']),
                    })

            # Process images
            images = []
            for source in filter(lambda x: x['type'] == 'image', sources):
                try:
                    img = Image.open(source['path'])
                    images.append({
                        "paper_id": source['paper_id'],
                        "path": source['path'],
                        "image": img
                    })
                except Exception as e:
                    logger.error(f"Error loading image {source['path']}: {e}")

            # Clean and post-process the answer
            # answer = self.clean_answer(results['result'])
            answer = results['result']

            logger.info("Query processing completed")
            return {
                "answer" : answer,
                "sources": sources,
                "images" : images
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer" : f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "images" : []
            }

