# Import required libraries
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch
import faiss
import json
import os
import re
from langchain.chains import RetrievalQA
from PIL import Image
from IPython.display import display
from typing import List
from pydantic import BaseModel, Field
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings