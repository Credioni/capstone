#pylint: disable=all
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
import os
import cv2
import faiss
import torch
import librosa
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

@dataclass
class TextMetadata:
    id: str
    title: str = ""
    source: str = ""
    text: str = ""


@dataclass
class ImageMetadata:
    id: str
    path: str
    caption: str = ""
    source: str = ""


@dataclass
class VideoMetadata:
    video_id: str
    title: str = ""
    description: str = ""
    author: str = ""
    duration_seconds: int = 0
    transcript: str = ""
    has_transcript: bool = False
    error: str = ""


@dataclass
class AudioMetadata:
    id: str
    path: str
    duration_seconds: float = 0
    sample_rate: int = 0
    transcript: str = ""
    source: str = ""
    error: str = ""




@dataclass
class MultimodalEmbedder:
    #text_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    whisper_model_name: str = "openai/whisper-tiny.en"

    # Fields initialized in __post_init__
    text_dim:  int = field(init=False)
    image_dim: int = field(init=False)
    video_dim: int = field(init=False)
    audio_dim: int = field(init=False)

    text_index:  Any = field(init=False, default=None)
    image_index: Any = field(init=False, default=None)
    video_index: Any = field(init=False, default=None)
    audio_index: Any = field(init=False, default=None)

    text_tokenizer: Any = field(init=False, default=None)
    text_model: Any = field(init=False, default=None)
    clip_processor: Any = field(init=False, default=None)
    clip_model: Any = field(init=False, default=None)
    whisper_processor: Any = field(init=False, default=None)
    whisper_model: Any = field(init=False, default=None)

    device: Any = field(init=False, default=None)

    text_metadata:  List[TextMetadata]  = field(default_factory=list)
    image_metadata: List[ImageMetadata] = field(default_factory=list)
    video_metadata: List[VideoMetadata] = field(default_factory=list)
    audio_metadata: List[AudioMetadata] = field(default_factory=list)

    def __post_init__(self):
        """Dataclasses post_init"""
        # Initialize dimensions
        self.text_dim  = 384  # Dimension of text embeddings
        self.audio_dim = 384  # Dimension same as text embeddings -transcriptions.
        self.image_dim = 512  # Dimension of CLIP visual embeddings
        self.video_dim = 512  # Dimension same as image dimension

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        print("Loading text embedding model...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.text_model.to(self.device)

        print("Loading image/video embedding model...")
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_model.to(self.device)

        print("Loading audio transcription model...")
        self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name)
        self.whisper_model.to(self.device)
        # Create FAISS indices
        self.setup_indices()


    def search(self, query:Dict[str, Any], k:int = 5, modality:str = "all") -> Dict[str, List]:
        """
        Search indexed content based on a text query
        modality: 'all', 'text', 'image', 'video', or 'audio'
        """
        results = {}

        if modality in ["all", "text"] and (query_text := query.get("text")) is not None:
            text_embedding = self.get_text_embedding(query_text)
        else:
            text_embedding = None

        if modality in ["all", "text"] and text_embedding is not None:
            # Search text index
            # query_embedding = self.get_text_embedding(query_text)

            scores, indices = self.text_index.search(text_embedding, k)
            text_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.text_metadata):
                    # Convert dataclass to dict and add score
                    result_dict = vars(self.text_metadata[idx])
                    result_dict["score"] = float(score)
                    text_results.append(result_dict)
            results["text"] = text_results

        if modality in ["all", "image"]:
            # Convert query to CLIP text embedding for image search
            # image = cv2.imread(query_imagepath)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                inputs = self.clip_processor(text=query["text"], return_tensors="pt", padding=True).to(self.device)
                #inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                query_img_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy()
                query_img_embedding = query_img_embedding / np.linalg.norm(query_img_embedding)

            scores, indices = self.image_index.search(query_img_embedding, k)
            image_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.image_metadata):
                    # Convert dataclass to dict and add score
                    result_dict = vars(self.image_metadata[idx])
                    result_dict["score"] = float(score)
                    image_results.append(result_dict)
            results["image"] = image_results


        if modality in ["all", "video"]:
            # Use same CLIP text embedding for video search
            with torch.no_grad():
                inputs = self.clip_processor(
                    text=query["text"],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                query_vid_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy()
                query_vid_embedding = query_vid_embedding / np.linalg.norm(query_vid_embedding)

            scores, indices = self.video_index.search(query_vid_embedding, k)
            video_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.video_metadata):
                    # Convert dataclass to dict and add score
                    result_dict = vars(self.video_metadata[idx])
                    result_dict["score"] = float(score)
                    video_results.append(result_dict)
            results["video"] = video_results


        if modality in ["all", "audio"] and text_embedding is not None:
            # Search audio index using the same text embeddings
            scores, indices = self.audio_index.search(text_embedding, k)
            audio_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.audio_metadata):
                    # Convert dataclass to dict and add score
                    result_dict = vars(self.audio_metadata[idx])
                    result_dict["score"] = float(score)
                    audio_results.append(result_dict)
            results["audio"] = audio_results

        return results

    def setup_indices(self):
        """Create separate FAISS indices for each modality"""
        # Using IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.text_index  = faiss.IndexFlatIP(self.text_dim)
        self.image_index = faiss.IndexFlatIP(self.image_dim)
        self.video_index = faiss.IndexFlatIP(self.video_dim)
        self.audio_index = faiss.IndexFlatIP(self.audio_dim)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text using the text embedding model"""
        inputs = self.text_tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # Use mean pooling to get a single vector per text
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embeddings for an image using CLIP"""
        try:
            # Load and process image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)

            # Get image embeddings and normalize
            embeddings = outputs.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros((1, self.image_dim))

    def get_youtube_video_embedding(self, video_id:str, num_frames:int = 5) -> Tuple[np.ndarray, VideoMetadata]:
        """ Generate embeddings for a YouTube video by:
            1. Downloading the video
            2. Extracting frames at regular intervals
            3. Getting CLIP embeddings for each frame
            4. Averaging them into a single embedding
            5. Getting transcript if available
        """
        # try:
        # Create a temporary directory for the video
        video_folderpath = os.path.join("embedding_data", "video")
        os.makedirs(video_folderpath, exist_ok=True)
        # Initialize YouTube object regardless of whether we need to download
        video_path = os.path.join(video_folderpath, f"{video_id}.mp4")
        video_name = f"https://www.youtube.com/watch?v={video_id}"
        # print(f"{video_name = }")
        yt = YouTube(str(video_name))

        # Store metadata
        metadata = VideoMetadata(
            video_id=video_id,
            title=yt.title,
            description=yt.description,
            author=yt.author,
            duration_seconds=yt.length
        )

        # Download the video if not already downloaded
        if not os.path.exists(video_path):
            logger.info(f"Downloading youtube video {video_id}")
            stream = yt.streams.filter(progressive=True, file_extension='.mp4').first()
            stream.download(output_path=video_folderpath, filename=f"{video_id}.mp4")

        # Get transcript if available
        transcript_text = ""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item['text'] for item in transcript])
            metadata.transcript = transcript_text
            metadata.has_transcript = True
        except Exception as e:
            print(f"No transcript available for {video_id}: {e}")

        # Process the video to extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract frames at regular intervals
        frame_embeddings = []
        interval = total_frames // (num_frames + 1)

        for i in range(1, num_frames + 1):
            frame_position = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()

            if ret:
                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get CLIP embedding for the frame
                inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.clip_model.get_image_features(**inputs)

                frame_embedding = outputs.cpu().numpy()
                frame_embeddings.append(frame_embedding)

        cap.release()

        # Process embeddings
        if frame_embeddings:
            # Average the frame embeddings
            visual_embedding = np.mean(np.vstack(frame_embeddings), axis=0, keepdims=True)
            # Normalize the embedding
            final_embedding = visual_embedding / np.linalg.norm(visual_embedding)
            return final_embedding, metadata

        # Fallback if no frames were processed
        return np.zeros((1, self.video_dim)), metadata

        # except Exception as e:
        #     print(f"Error processing video {video_id}: {e}")
        #     error_metadata = VideoMetadata(video_id=video_id, error=str(e))
        #     return np.zeros((1, self.video_dim)), error_metadata

    def index_documents(self, documents: List[Dict[str, str]]):
        """Index text documents with metadata"""
        for doc in tqdm(documents, desc="Indexing documents"):
            embedding = self.get_text_embedding(doc["text"])
            self.text_index.add(embedding)

            # Create metadata and add truncated text
            text_content = doc.get("text", "")
            truncated_text = text_content[:300] + "..." if len(text_content) > 300 else text_content

            metadata = TextMetadata(
                id=doc.get("id", str(len(self.text_metadata))),
                title=doc.get("title", ""),
                source=doc.get("source", ""),
                text=truncated_text
            )
            self.text_metadata.append(metadata)

    def index_images(self, image_paths: List[Dict[str, str]]):
        """Index images with metadata"""
        for img in tqdm(image_paths, desc="Indexing images"):
            embedding = self.get_image_embedding(img["path"])
            self.image_index.add(embedding)

            metadata = ImageMetadata(
                id=img.get("id", str(len(self.image_metadata))),
                path=img["path"],
                caption=img.get("caption", ""),
                source=img.get("source", "")
            )
            self.image_metadata.append(metadata)

    def index_audio_files(self, audio_paths: List[str]):
        """Index audio files with metadata"""
        for audio_path in tqdm(audio_paths, desc="Indexing audio files"):
            embedding, metadata = self.get_audio_embedding(audio_path)
            self.audio_index.add(embedding)
            self.audio_metadata.append(metadata)

    def index_youtube_videos(self, video_ids: List[str]):
        """Index YouTube videos with metadata"""
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            embedding, metadata = self.get_youtube_video_embedding(video_id)
            self.video_index.add(embedding)
            self.video_metadata.append(metadata)

    def get_audio_embedding(self, audio_path: str) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Generate embeddings for an audio file by:
        1. Loading the audio file
        2. Transcribing it using Whisper
        3. Getting text embeddings for the transcription

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of audio embedding and metadata
        """
        try:
            # Create metadata object
            metadata = AudioMetadata(
                id=os.path.basename(audio_path),
                path=audio_path
            )

            # Load audio using librosa
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            metadata.sample_rate = sample_rate
            metadata.duration_seconds = len(audio_array) / sample_rate

            # Convert to format expected by Whisper
            inputs = self.whisper_processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)

            # Generate transcription
            with torch.no_grad():
                outputs = self.whisper_model.generate(**inputs, max_length=448)
                transcription = self.whisper_processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0]

            metadata.transcript = transcription

            # Get text embedding of the transcription
            text_embedding = self.get_text_embedding(transcription)

            return text_embedding, metadata

        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            error_metadata = AudioMetadata(
                id=os.path.basename(audio_path),
                path=audio_path,
                error=str(e)
            )
            return np.zeros((1, self.audio_dim)), error_metadata

    def save_indices_images(self, directory:str="faiss_indices"):
        """Save FAISS indices and metadata to disk"""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.image_index, f"{directory}/image_index.faiss")
        image_metadata_dicts = [vars(item) for item in self.image_metadata]
        pd.DataFrame(image_metadata_dicts).to_json(f"{directory}/image_metadata.json", orient="records")

    def save_indices_audio(self, directory:str="faiss_indices"):
        """Save FAISS indices and metadata to disk"""
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.audio_index, f"{directory}/audio_index.faiss")
        audio_metadata_dicts = [vars(item) for item in self.audio_metadata]
        pd.DataFrame(audio_metadata_dicts).to_json(f"{directory}/audio_metadata.json", orient="records")

    def save_indices_video(self, directory:str="faiss_indices"):
        """Save FAISS indices and metadata to disk"""
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.video_index, f"{directory}/video_index.faiss")
        video_metadata_dicts = [vars(item) for item in self.video_metadata]
        pd.DataFrame(video_metadata_dicts).to_json(f"{directory}/video_metadata.json", orient="records")

    def save_indices_all(self, directory:str = "faiss_indices"):
        """Save FAISS indices and metadata to disk"""
        os.makedirs(directory, exist_ok=True)

        # # Save FAISS indices
        faiss.write_index(self.text_index, f"{directory}/text_index.faiss")
        faiss.write_index(self.image_index, f"{directory}/image_index.faiss")
        faiss.write_index(self.audio_index, f"{directory}/audio_index.faiss")
        faiss.write_index(self.video_index, f"{directory}/video_index.faiss")

        # Convert dataclasses to dicts for serialization
        text_metadata_dicts = [vars(item) for item in self.text_metadata]
        image_metadata_dicts = [vars(item) for item in self.image_metadata]
        audio_metadata_dicts = [vars(item) for item in self.audio_metadata]
        video_metadata_dicts = [vars(item) for item in self.video_metadata]

        # Save metadata
        pd.DataFrame(text_metadata_dicts).to_json(f"{directory}/text_metadata.json", orient="records")
        pd.DataFrame(image_metadata_dicts).to_json(f"{directory}/image_metadata.json", orient="records")
        pd.DataFrame(audio_metadata_dicts).to_json(f"{directory}/audio_metadata.json", orient="records")
        pd.DataFrame(video_metadata_dicts).to_json(f"{directory}/video_metadata.json", orient="records")

    def load_indices(self, directory: str = "faiss_indices"):
        """Load FAISS indices and metadata from disk"""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        current_dir = os.path.join(dir_path, directory)

        # Load FAISS indices
        self.text_index = faiss.read_index(os.path.join(current_dir, "text_index.faiss"))
        self.image_index = faiss.read_index(os.path.join(current_dir,"image_index.faiss"))
        self.video_index = faiss.read_index(os.path.join(current_dir, "video_index.faiss"))
        self.audio_index = faiss.read_index(os.path.join(current_dir, "audio_index.faiss"))

        # Load metadata and convert to dataclasses
        text_data = pd.read_json(os.path.join(current_dir, "text_metadata.json"), orient="records").to_dict("records")
        self.text_metadata = [TextMetadata(**item) for item in text_data]

        image_data = pd.read_json(os.path.join(current_dir, "image_metadata.json"), orient="records").to_dict("records")
        self.image_metadata = [ImageMetadata(**item) for item in image_data]

        video_data = pd.read_json(os.path.join(current_dir, "video_metadata.json"), orient="records").to_dict("records")
        self.video_metadata = [VideoMetadata(**item) for item in video_data]

        audio_data = pd.read_json(os.path.join(current_dir, "audio_metadata.json"), orient="records").to_dict("records")
        self.audio_metadata = [AudioMetadata(**item) for item in audio_data]




if __name__=="__main__":
    def main():
        embedder = MultimodalEmbedder()
        # Example text documents (e.g., from arXiv)
        documents = [
            {
                "id": "2104.08663",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "source": "arXiv",
                "text": "Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remains challenging. Pre-trained models with a differentiable access mechanism to explicit nonparametric memory can overcome this limitation, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the entire generated text, and another which can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline."
            },
            {
                "id": "2103.00020",
                "title": "CLIP: Connecting Text and Images",
                "source": "arXiv",
                "text": "We present a neural network that efficiently learns visual concepts from natural language supervision. Our method can be applied to any visual concept that people can describe in language and is trained using pairs of images and text found across the internet. By design, the network can be instructed in natural language to perform a wide variety of classification benchmarks, without directly optimizing for the benchmark's performance, similar to the \"zero-shot\" capabilities of GPT-2 and 3. We find that this approach is efficient and scalable, achieving good performance on a variety of image classification datasets. It also enables flexible zero-shot transfer, where a single model can be adapted to perform many different tasks."
            }
        ]

        # Example images (paths would be different in your system)
        images = [
            {
                "id": 1,
                "path": "tmp/image1.png",
                "caption": "Diagram of a Retrieval Augmented Generation system",
                "source": "Research paper figure"
            },
        ]

        # Example YouTube videos related to academic research
        youtube_videos = [
            "eMlx5fFNoYc",  # Attention in transformers, step-by-step | 3B1B
        ]

        # Index content
        embedder.index_documents(documents)
        embedder.index_images(images)
        embedder.index_youtube_videos(youtube_videos)
        embedder.save_indices(os.path.join("tmp", "faiss_indices"))

        # Example search
        print("Example search:")
        print("To run a search, load the indices first or index some content")
        print("results = embedder.search('multimodal embeddings for scientific papers', k=3)")

        results = embedder.search('CLIP: Connecting Text and Images', k=3)
        print(results.items())
        print(*results["text"], sep="\n")

    main()
