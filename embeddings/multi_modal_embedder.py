#
from typing import List, Dict, Tuple
import os
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from pytubefix import YouTube
import cv2
from youtube_transcript_api import YouTubeTranscriptApi

import pandas as pd
from tqdm import tqdm

class MultimodalEmbedder:
    def __init__(self):
        # Initialize models
        print("Loading text embedding model...")
        self.text_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.text_dim = 768  # Dimension of text embeddings

        print("Loading image/video embedding model...")
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.image_dim = 512  # Dimension of CLIP visual embeddings

        # For video, we'll use the same CLIP model but process multiple frames
        self.video_dim = 512  # Same as image dimension

        # Create FAISS indices
        self.setup_indices()

        # Storage for metadata
        self.text_metadata = []
        self.image_metadata = []
        self.video_metadata = []

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model.to(self.device)
        self.clip_model.to(self.device)

    def setup_indices(self):
        """Create separate FAISS indices for each modality"""
        # Using IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.text_index = faiss.IndexFlatIP(self.text_dim)
        self.image_index = faiss.IndexFlatIP(self.image_dim)
        self.video_index = faiss.IndexFlatIP(self.video_dim)

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

    def get_youtube_video_embedding(self, video_id: str, num_frames: int = 5) -> Tuple[np.ndarray, dict]:
        """
        Generate embeddings for a YouTube video by:
        1. Downloading the video
        2. Extracting frames at regular intervals
        3. Getting CLIP embeddings for each frame
        4. Averaging them into a single embedding
        5. Getting transcript if available
        """
        try:
            # Create a temporary directory for the video
            os.makedirs("temp", exist_ok=True)
            video_path = f"temp/{video_id}.mp4"

            # Initialize YouTube object regardless of whether we need to download
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

            # Store metadata
            metadata = {
                "video_id": video_id,
                "title": yt.title,
                "description": yt.description,
                "author": yt.author,
                "duration_seconds": yt.length
            }

            # Download the video if not already downloaded
            if not os.path.exists(video_path):
                stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
                stream.download(output_path="temp", filename=f"{video_id}.mp4")

            # Get transcript if available
            transcript_text = ""
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([item['text'] for item in transcript])
                metadata["transcript"] = transcript_text
            except Exception as e:
                print(f"No transcript available for {video_id}: {e}")
                metadata["transcript"] = ""

            # Process the video to extract frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps

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

            # Combine frame embeddings with transcript embedding if available
            if frame_embeddings:
                # Average the frame embeddings
                visual_embedding = np.mean(np.vstack(frame_embeddings), axis=0, keepdims=True)

                # If transcript is available, process separately but don't try to combine embeddings of different dimensions
                if transcript_text:
                    # Store transcript in metadata but don't try to combine embeddings with different dimensions
                    metadata["has_transcript"] = True
                    metadata["transcript"] = transcript_text

                    # Just use the visual embedding for the index
                    final_embedding = visual_embedding / np.linalg.norm(visual_embedding)
                else:
                    final_embedding = visual_embedding / np.linalg.norm(visual_embedding)

                return final_embedding, metadata

            # Fallback if no frames were processed
            return np.zeros((1, self.video_dim)), metadata

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            return np.zeros((1, self.video_dim)), {"video_id": video_id, "error": str(e)}

    def index_documents(self, documents: List[Dict[str, str]]):
        """Index text documents with metadata"""
        for doc in tqdm(documents, desc="Indexing documents"):
            embedding = self.get_text_embedding(doc["text"])
            self.text_index.add(embedding)
            self.text_metadata.append({
                "id": doc.get("id", len(self.text_metadata)),
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "text": doc.get("text", "")[:200] + "..."  # Store preview
            })

    def index_images(self, image_paths: List[Dict[str, str]]):
        """Index images with metadata"""
        for img in tqdm(image_paths, desc="Indexing images"):
            embedding = self.get_image_embedding(img["path"])
            self.image_index.add(embedding)
            self.image_metadata.append({
                "id": img.get("id", len(self.image_metadata)),
                "path": img["path"],
                "caption": img.get("caption", ""),
                "source": img.get("source", "")
            })

    def index_youtube_videos(self, video_ids: List[str]):
        """Index YouTube videos with metadata"""
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            embedding, metadata = self.get_youtube_video_embedding(video_id)
            self.video_index.add(embedding)
            self.video_metadata.append(metadata)

    def search(self, query: str, k: int = 5, modality: str = "all") -> Dict[str, List]:
        """
        Search indexed content based on a text query
        modality: 'all', 'text', 'image', or 'video'
        """
        query_embedding = self.get_text_embedding(query)
        results = {}

        if modality in ["all", "text"]:
            # Search text index
            scores, indices = self.text_index.search(query_embedding, k)
            results["text"] = [{**self.text_metadata[idx], "score": float(score)}
                              for score, idx in zip(scores[0], indices[0]) if idx >= 0]

        if modality in ["all", "image"]:
            # Convert query to CLIP text embedding for image search
            with torch.no_grad():
                inputs = self.clip_processor(text=query, return_tensors="pt", padding=True).to(self.device)
                query_img_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy()
                query_img_embedding = query_img_embedding / np.linalg.norm(query_img_embedding)

            scores, indices = self.image_index.search(query_img_embedding, k)
            results["image"] = [{**self.image_metadata[idx], "score": float(score)}
                               for score, idx in zip(scores[0], indices[0]) if idx >= 0]

        if modality in ["all", "video"]:
            # Use same CLIP text embedding for video search
            with torch.no_grad():
                inputs = self.clip_processor(text=query, return_tensors="pt", padding=True).to(self.device)
                query_vid_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy()
                query_vid_embedding = query_vid_embedding / np.linalg.norm(query_vid_embedding)

            scores, indices = self.video_index.search(query_vid_embedding, k)
            results["video"] = [{**self.video_metadata[idx], "score": float(score)}
                               for score, idx in zip(scores[0], indices[0]) if idx >= 0]

        return results

    def save_indices(self, directory: str = "faiss_indices"):
        """Save FAISS indices and metadata to disk"""
        os.makedirs(directory, exist_ok=True)

        # Save FAISS indices
        faiss.write_index(self.text_index, f"{directory}/text_index.faiss")
        faiss.write_index(self.image_index, f"{directory}/image_index.faiss")
        faiss.write_index(self.video_index, f"{directory}/video_index.faiss")

        # Save metadata
        pd.DataFrame(self.text_metadata).to_json(f"{directory}/text_metadata.json", orient="records")
        pd.DataFrame(self.image_metadata).to_json(f"{directory}/image_metadata.json", orient="records")
        pd.DataFrame(self.video_metadata).to_json(f"{directory}/video_metadata.json", orient="records")

    def load_indices(self, directory: str = "faiss_indices"):
        """Load FAISS indices and metadata from disk"""
        # Load FAISS indices
        self.text_index = faiss.read_index(f"{directory}/text_index.faiss")
        self.image_index = faiss.read_index(f"{directory}/image_index.faiss")
        self.video_index = faiss.read_index(f"{directory}/video_index.faiss")

        # Load metadata
        self.text_metadata = pd.read_json(f"{directory}/text_metadata.json", orient="records").to_dict("records")
        self.image_metadata = pd.read_json(f"{directory}/image_metadata.json", orient="records").to_dict("records")
        self.video_metadata = pd.read_json(f"{directory}/video_metadata.json", orient="records").to_dict("records")