#pylint: disable=all
import json
import logging
import glob
import traceback
import torch
from typing import Dict, Optional
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .multi_modal_embedder import MultimodalEmbedder

# Set up logging
logger = logging.getLogger(__name__)

class ArXivRAGSystem:
    """
    An enhanced RAG (Retrieval-Augmented Generation) system for arXiv papers
    and multi-media scientific content with improved prompt engineering and
    error handling.
    """

    def __init__(self, _config: Dict[str, str]=None):
        """
        Initialize the RAG system with the given configuration.

        Args:
            config: A dictionary containing configuration parameters:
                - faiss_index_path: Path to the FAISS index file
                - mapping_path: Path to the document mapping file
                - projection_path: Path to the projection model file
                - image_folder: Path to the folder containing images
                - use_mm_embedder: Whether to use the multimodal embedder (default: False)
        """
        #self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize components
        # if self.config.get("use_mm_embedder", False):
        #     logger.info("Using Multimodal Embedder")
        #     self.mm_embedder = self._initialize_mm_embedder()
        #     self.retriever = None
        # else:
        #     logger.info("Using ArXiv Retriever")
        #     self.mm_embedder = None
        #     self.retriever = self._initialize_retriever()

        self.llm = self._initialize_llm()
        # self.qa_chain = self._create_qa_chain()

    def _initialize_mm_embedder(self) -> MultimodalEmbedder:
        """Initialize the multimodal embedder."""
        try:
            embedder = MultimodalEmbedder()
            embedder.load_indices()
            logger.info("Initialized Multimodal Embedder")
            return embedder
        except Exception as e:
            logger.error(f"Failed to initialize multimodal embedder: {e}")
            raise

    # def _initialize_retriever(self) -> ArXivRetriever:
    #     """Initialize the arXiv document retriever."""
    #     try:
    #         retriever = ArXivRetriever(
    #             index_path=self.config['faiss_index_path'],
    #             mapping_path=self.config['mapping_path'],
    #             projection_path=self.config['projection_path'],
    #             image_folder=self.config['image_folder'],
    #             device=self.device
    #         )
    #         logger.info("Initialized ArXiv retriever")
    #         return retriever
    #     except Exception as e:
    #         logger.error(f"Failed to initialize retriever: {e}")
    #         raise

    def _initialize_llm(self):
        """Initialize the language model with optimized settings."""
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                # Add low_cpu_mem_usage for better performance on limited resources
                low_cpu_mem_usage=True
            )

            # Optimized generation parameters for DeepSeek models
            text_generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,  # Increased for more detailed answers
                temperature=0.5,  # Reduced for more focused responses
                top_p=0.95,  # Slightly increased for more creativity while staying factual
                repetition_penalty=1.15,  # Slightly increased to avoid repetition
                do_sample=True,
                # Added stopping criteria for better generation
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

            llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info(f"Initialized language model: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise


    def find_document_path(self, paper_id: str) -> Optional[str]:
        """
        Find the document path for a given paper ID.

        Args:
            paper_id: The ID of the paper

        Returns:
            The path to the PDF file or None if not found
        """
        try:
            pdf_files = glob.glob(f"data/pdfs/{paper_id}.pdf")
            if not pdf_files:
                return None
            return pdf_files[0]
        except Exception as e:
            logger.error(f"Error finding document path: {e}")
            return None

    def clean_answer(self, text: str) -> str:
        """
        Clean the generated answer by removing duplicate content and fixing formatting.
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

        # Join sentences back together
        cleaned_text = '. '.join(unique)
        # Fix any potential extra spaces or artifacts
        cleaned_text = cleaned_text.replace('  ', ' ').strip()

        return cleaned_text

    # def mm_query(self, question: str, k: int = 5) -> Dict[str, any]:
    #     """
    #     Query using the multimodal embedder.

    #     Args:
    #         question: The question to answer
    #         k: Number of documents to retrieve

    #     Returns:
    #         A dictionary containing answer and sources
    #     """
    #     logger.info(f"Processing multimodal query: {question}")

    #     try:
    #         # Query using the multimodal embedder
    #         search_results = self.mm_embedder.search({"text": question}, k=k)

    #         # Prepare context from search results
    #         context_parts = []
    #         sources = []
    #         images = []

    #         # Process text results
    #         if "text" in search_results:
    #             for item in search_results["text"]:
    #                 context_parts.append(f"TEXT CONTENT (score={item['score']:.2f}):\n{item['text']}")
    #                 sources.append({
    #                     "type": "text",
    #                     "score": item["score"],
    #                     "content": item["text"],
    #                     "title": item.get("title", ""),
    #                     "source": item.get("source", "")
    #                 })

    #         # Process image results
    #         if "image" in search_results:
    #             for item in search_results["image"]:
    #                 context_parts.append(f"IMAGE CAPTION (score={item['score']:.2f}):\n{item['caption']}")
    #                 sources.append({
    #                     "type": "image",
    #                     "score": item["score"],
    #                     "path": item["path"],
    #                     "caption": item["caption"]
    #                 })
    #                 try:
    #                     images.append({
    #                         "path": item["path"],
    #                         "caption": item["caption"],
    #                         "image": Image.open(item["path"])
    #                     })
    #                 except Exception as e:
    #                     logger.error(f"Error loading image {item['path']}: {e}")

    #         # Process video results
    #         if "video" in search_results:
    #             for item in search_results["video"]:
    #                 if item.get("transcript"):
    #                     context_parts.append(f"VIDEO TRANSCRIPT (score={item['score']:.2f}):\n{item['transcript'][:500]}...")
    #                 else:
    #                     context_parts.append(f"VIDEO TITLE (score={item['score']:.2f}):\n{item['title']}")

    #                 sources.append({
    #                     "type": "video",
    #                     "score": item["score"],
    #                     "video_id": item["video_id"],
    #                     "title": item.get("title", ""),
    #                     "transcript": item.get("transcript", "")[:500] + "..." if item.get("transcript") else ""
    #                 })

    #         # Create full context
    #         context = "\n\n---\n\n".join(context_parts)

    #         # Generate response using the LLM
    #         # Prepare the prompt
    #         prompt = self._prepare_prompt(context, question)

    #         # Generate using the LLM pipeline
    #         inputs = {
    #             "text": prompt,
    #             "max_new_tokens": 512,
    #             "do_sample": True,
    #             "temperature": 0.5,
    #             "top_p": 0.95
    #         }

    #         generated_response = self.llm(inputs)
    #         answer = self.clean_answer(generated_response[0]['generated_text'])

    #         logger.info("Multimodal query processing completed")
    #         return {
    #             "answer": answer,
    #             "sources": sources,
    #             "images": images
    #         }

    #     except Exception as e:
    #         logger.error(f"Error processing multimodal query: {e}")
    #         return {
    #             "answer": f"An error occurred while processing your query: {str(e)}",
    #             "sources": [],
    #             "images": []
    #         }

    def _prepare_prompt(self, context:list, question: str) -> str:
        """
        Prepare the prompt for the LLM.

        Args:
            context: The context information
            question: The question to answer

        Returns:
            The formatted prompt
        """
        context = "\n\n" + "\n\n".join(context) + "\n\n"
        return f"""<|im_start|>system
You are a helpful, accurate, and concise research assistant specialized in scientific knowledge from arXiv papers. Your goal is to provide factual, well-structured answers based on the information provided.
<|im_end|>
<|im_start|>user
I need a detailed and accurate answer to a scientific question. Please use only the provided context and don't make up information.
Also reference to only one article based on provided context by it's number and title as you see best.
CONTEXT_START
{context}
CONTEXT_END
QUESTION:
{question}
<|im_end|>
<|im_start|>assistant
I'll answer the question based on the provided context, between the CONTEXT_START and CONTEXT_END.
Each article is separated with number, followed by title, and next the abstract of the article.

</think>
"""

    def query(self, question:str, context:list, k:int = 5, score_threshold:float = 0.6) -> Dict:
        """
        Query the RAG system with a question and optional context.

        Args:
            question: The question to answer
            context: Optional pre-defined context to use instead of or in addition to retrieved content
            k: Number of documents to retrieve
            score_threshold: Threshold for filtering documents by score

        Returns:
            A dictionary containing:
                - answer: The generated answer
                - sources: The source documents used for the answer
                - images: Any images referenced in the sources
        """
        logger.info(f"Processing query: {question}")
        logger.debug(f"Context provided:\n{context}")

        try:
            # Create a prompt with the provided context
            prompt = self._prepare_prompt(context, question)

            # Generate using the LLM pipeline
            generated_response = self.llm.invoke(prompt, max_new_tokens=512)
            # answer = self.clean_answer(generated_response[0]['generated_text'])
            logger.info("Generation completed processing completed")
            logger.info(json.dumps(generated_response, indent=2))

            return generated_response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            return None