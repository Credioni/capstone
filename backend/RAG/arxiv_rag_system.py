#pylint: disable=E0110,E0611
import os
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .arxiv_retriever import ArXivRetriever
import glob

torch.set_default_device("cuda")


# RAG System Class
class ArXivRAGSystem:
    """ Init """

    def __init__(self, config):
        """ Init """
        self.config = config
        self.retriever = ArXivRetriever(
            index_path=config['faiss_index_path'],
            mapping_path=config['mapping_path'],
            projection_path=config['projection_path'],
            image_folder=config['image_folder']
        )
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()

    def _initialize_llm(self):
        # self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.model_name = "microsoft/phi-2"  # Requires ~5GB VRAM
        # self.model_name = "gpt2"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            #device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )#.to("cuda")

        return HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        ))

    def _create_qa_chain(self):
        """ Init """

        prompt_template = """Generate a detailed answer to the question below using the provided context.
        Include references to figures and papers where applicable.
        Context: {context}

        Question: {question}

        Structured Answer:"""

        return RetrievalQA.from_chain_type(
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

    def query(self, question, k=5, score_threshold=0.6):
        """ Init """
        results = self.qa_chain.invoke({"query": question, "k": k})

        # Debug: Print the full prompt sent to self.llm
        print("\n=== DEBUG: FULL PROMPT ===")
        print(f"Context: {results['source_documents']}")
        print(f"Question: {question}")
        print("==========================\n")
        print("Context being passed to the prompt:")

        for doc in results['source_documents']:
            print(f"Document: {doc.page_content[:200]}... (Score: {doc.metadata['score']})")

        print("Prossessing sources...")
        # Process sources (LOWER distance = BETTER match)
        sources = []
        for doc in results['source_documents']:
            if doc.metadata['score'] < score_threshold:  # Keep good matches
                sources.append({
                    "type": doc.metadata['type'],
                    "paper_id": doc.metadata['paper_id'],
                    "score": doc.metadata['score'],
                    "content": doc.page_content,
                    # "path": doc.metadata['path'],
                    "path": self.doc_path(doc.metadata['paper_id']),
                })

        # Process images
        images = []
        for source in sources:
            if source['type'] == 'image' and os.path.exists(source['path']):
                try:
                    img = Image.open(source['path'])
                    images.append({
                        "paper_id": source['paper_id'],
                        "path": source['path'],
                        "image": img
                    })
                except Exception as e:
                    print(f"Error loading image {source['path']}: {str(e)}")

        # Post-process answer
        answer = results['result'] #self._clean_answer(results['result'])

        print("Quering completed.")
        return {
            "answer": answer,
            "sources": sources,
            "images": images
        }

    def doc_path(self, id):
        """Find doct path in data with given id"""
        print("current path", os.getcwd())
        pdf_file = glob.glob(f"data/pdfs/{id}.pdf")[0]
        return pdf_file

    def _clean_answer(self, text):
        """ Init """
        # Remove duplicate sentences
        sentences = text.split('. ')
        seen = set()
        unique = []
        for sent in sentences:
            key = sent[:50].lower()  # Check first 50 chars for duplicates
            if key not in seen:
                seen.add(key)
                unique.append(sent)

        return '. '.join(unique)