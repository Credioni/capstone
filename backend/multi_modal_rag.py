#pylint: disable=all
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch

class Storages:
    def __init__(self):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.audio_vectorstore = None


class MultimodalRAG:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize language model
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Initialize text embeddings
        self.text_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize image embeddings
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        # Initialize Audio transcript model
        processor = AutoProcessor.from_pretrained("openai/whisper-small")
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-small",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            low_cpu_mem_usage=True
        )

        self.whisper_audio_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model.to(self.device),
            tokenizer=processor
        )


        # Vector stores

