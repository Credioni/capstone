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
    score: float = 0.0


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

    text_metadata: List[TextMetadata] = field(default_factory=list)
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

        # Add projection layers to map between embedding spaces
        # These help with cross-modal search between different modalities
        self.text_to_image_projection = torch.nn.Linear(self.text_dim, self.image_dim)
        self.text_to_image_projection.to(self.device)

        self.image_to_text_projection = torch.nn.Linear(self.image_dim, self.text_dim)
        self.image_to_text_projection.to(self.device)

        # Create FAISS indices
        self.setup_indices()

    def search(self, query: Dict[str, Any], k: int = 5) -> dict:
        """Search indexed content based on multiple query types (text, image, audio).
        Each query type can search across all modalities where applicable.

        Args:
            query (Dict[str, Any]): Dictionary containing query information:
                - text: Text query string (optional)
                - image: Path to image file for image query (optional)
                - audio: Path to audio file for audio query (optional)
            k: Number of results to return per modality. Defaults to 5.

        Returns:
            dict: Dict with results organized by modality (text, image, video, audio).
        """
        results = {}
        text_embedding = None
        image_embedding = None
        audio_embedding = None
        image_for_text_embedding = None  # Specialized embedding for image-to-text search

        ######## Process all provided query types to generate embeddings #########
        # Process text query
        if query_text := query.get("text"):
            text_embedding = self.get_text_embedding(query_text)

        # Process image query - now with image-to-text option
        if image_path := query.get("image"):
            try:
                # Standard image embedding for image-to-image search
                image_embedding = self.get_image_embedding(image_path)

                # Specialized embedding for image-to-text search
                image_for_text_embedding = self.get_image_to_text_embedding(image_path)

            except Exception as e:
                logger.error(f"Error processing image query: {e}")

        # Process audio query
        if audio_path := query.get("audio"):
            logger.info(f"{audio_path = }")
            try:
                # Get transcript from audio
                transcript = self.get_transcript_from_audio(audio_path)

                # Generate text embedding from transcript
                audio_embedding = self.get_text_embedding(transcript)

                # Store transcript in results for user context
                results["audio_transcript"] = transcript
            except Exception as e:
                logger.error(f"Error processing audio query: {e}")

        ###### Search all modalities using available embeddings #####

        # Search text index - now supporting image-to-text search
        if text_embedding is not None or audio_embedding is not None or image_for_text_embedding is not None:
            # Determine which embedding to use (prioritize in this order: text, image-for-text, audio)
            if text_embedding is not None:
                search_embedding = text_embedding
            elif image_for_text_embedding is not None:
                search_embedding = image_for_text_embedding
            else:
                search_embedding = audio_embedding

            results["text"] = self.search_modality(search_embedding, self.text_index, self.text_metadata, k)

        # Search image index
        if text_embedding is not None or image_embedding is not None:
            # For text-to-image search, convert text embedding to image space using CLIP
            if text_embedding is not None and image_embedding is None:
                search_embedding = self.get_text_to_image_embedding(query["text"])
            else:
                search_embedding = image_embedding

            results["image"] = self.search_modality(search_embedding, self.image_index, self.image_metadata, k)

        # Search video index - similar approach to images
        if text_embedding is not None or image_embedding is not None:
            # For text-to-video search, same as text-to-image
            if text_embedding is not None and image_embedding is None:
                search_embedding = self.get_text_to_image_embedding(query["text"])
            else:
                search_embedding = image_embedding

            results["video"] = self.search_modality(search_embedding, self.video_index, self.video_metadata, k)

        # Search audio index - Use text embedding or image-to-text embedding for searching
        if text_embedding is not None or audio_embedding is not None or image_for_text_embedding is not None:
            if text_embedding is not None:
                search_embedding = text_embedding
            elif image_for_text_embedding is not None:
                search_embedding = image_for_text_embedding
            else:
                search_embedding = audio_embedding

            results["audio"] = self.search_modality(search_embedding, self.audio_index, self.audio_metadata, k)

        return results

    def search_modality(self, query_embedding: np.ndarray, modality_index, modality_metadata, k: int = 5):
        """
        Generic search method for any modality.

        Args:
            query_embedding: The query embedding vector
            modality_index: FAISS index for the modality
            modality_metadata: List of metadata for the modality
            k: Number of results to return

        Returns:
            List of results with scores
        """
        scores, indices = modality_index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(modality_metadata):
                result_dict = vars(modality_metadata[idx])
                result_dict["score"] = float(score)
                results.append(result_dict)

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

    def get_text_to_image_embedding(self, text: str) -> np.ndarray:
        """
        Generate CLIP text embeddings that align with the image embedding space.
        This is specifically designed for text-to-image search.

        Args:
            text: The text query

        Returns:
            Embedding vector aligned with CLIP's visual space
        """
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            embeddings = text_features.cpu().numpy()

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def get_image_to_text_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate image embeddings that align well with the text embedding space.
        This is specifically designed for image-to-text search.

        Args:
            image_path: Path to the image file

        Returns:
            Embedding vector aligned with text space
        """
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Get image features from CLIP
                image_features = self.clip_model.get_image_features(**inputs)

                # Apply projection to align with text embedding space
                projected_features = self.image_to_text_projection(image_features)
                embeddings = projected_features.cpu().numpy()

            # Normalize for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

        except Exception as e:
            logger.error(f"Error processing image for text search: {e}")
            return np.zeros((1, self.text_dim))

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

    def get_transcript_from_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using Whisper model.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            inputs = self.whisper_processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.whisper_model.generate(**inputs, max_length=448)
                transcript = self.whisper_processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0]

            return transcript
        except Exception as e:
            logger.error(f"Error transcribing audio {audio_path}: {e}")
            return ""

    def get_youtube_video_embedding(self, video_id:str, num_frames:int = 5) -> Tuple[np.ndarray, VideoMetadata]:
        """ Generate embeddings for a YouTube video by:
            1. Downloading the video
            2. Extracting frames at regular intervals
            3. Getting CLIP embeddings for each frame
            4. Averaging them into a single embedding
            5. Getting transcript if available
        """
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

            # Get transcript
            transcript = self.get_transcript_from_audio(audio_path)
            metadata.transcript = transcript

            # Get text embedding of the transcription
            text_embedding = self.get_text_embedding(transcript)

            return text_embedding, metadata

        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            error_metadata = AudioMetadata(
                id=os.path.basename(audio_path),
                path=audio_path,
                error=str(e)
            )
            return np.zeros((1, self.audio_dim)), error_metadata

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

    def project_text_to_image_space(self, text_embedding: np.ndarray) -> np.ndarray:
        """
        Project text embeddings to image embedding space using the projection layer.

        Args:
            text_embedding: Text embeddings in original text space (384-dim)

        Returns:
            Projected embeddings in image space (512-dim)
        """
        try:
            # Convert numpy array to tensor
            tensor_embedding = torch.tensor(text_embedding).to(self.device)

            # Apply projection
            with torch.no_grad():
                projected = self.text_to_image_projection(tensor_embedding)

            # Convert back to numpy and normalize
            projected_np = projected.cpu().numpy()
            projected_np = projected_np / np.linalg.norm(projected_np, axis=1, keepdims=True)

            return projected_np
        except Exception as e:
            logger.error(f"Error projecting text to image space: {e}")
            return np.zeros((1, self.image_dim))

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



