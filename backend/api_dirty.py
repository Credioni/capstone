#pylint: disable=all
import os
import io
import cv2
import json
import base64
import traceback
import numpy as np
from PIL import Image
from enum import Enum
from typing import List

from RAGembedder.multi_modal_embedder import ImageMetadata


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # Return the value of the Enum
        return super().default(obj)



import logging
from logging_formatter import CustomFormatter

def init_logger(logger):
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)



def process_audio(audio_list, logger):
    """
    Process audio data for API response.

    Args:
        audio_list: List of audio objects from RAG system
        logger: Logger instance

    Returns:
        List of processed audio data with base64 encoding
    """
    def audio_to_base64(audio_path):
        try:
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                return audio_b64
        except Exception as e:
            return None

    processed_audio = []

    for audio_info in audio_list:
        try:
            # Get file extension from path
            _, ext = os.path.splitext(audio_info["path"])
            ext = ext.lstrip('.')

            audio_data = {
                "id": audio_info.get("id", ""),
                "path": audio_info["path"],
                "base64": audio_to_base64(os.path.join("RAGembedder", audio_info["path"])),
                # e.g., "audio/mp3", "audio/wav"
                "mime_type": f"audio/{ext}",
                "transcript": audio_info.get("transcript", ""),
                "duration_seconds": audio_info.get("duration_seconds", 0),
                "score": audio_info.get("score", 0),
            }
            processed_audio.append(audio_data)

            # logger.info(f"Processed audio file: {audio_info['path']}")
        except Exception as e:
            logger.error(f"Error processing audio {audio_info.get('path', 'unknown')}: {e}")
            logger.error(traceback.format_exc())

    return processed_audio

def process_images(images: List[ImageMetadata], logger):
    """ Process image data for API response """
    def image_to_base64(img):
        """ Convert a PIL Image to a base64-encoded string """
        buffered = io.BytesIO()

        # Convert OpenCV image (numpy array) to PIL Image
        if isinstance(img, np.ndarray):
            # OpenCV uses BGR, PIL uses RGB
            if img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            else:
                pil_img = Image.fromarray(img)
            pil_img.save(buffered, format="PNG")
        else:
            # Already a PIL Image
            img.save(buffered, format="PNG")

        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    processed_images = []
    for image in images:
        try:
            # Load the image with OpenCV
            img_path = os.path.join("RAGembedder", image["path"])
            cv_img = cv2.imread(img_path)

            if cv_img is None:
                logger.error(f"Failed to load image: {img_path}")
                continue

            # Get image dimensions
            height, width = cv_img.shape[:2]

            img_data = {
                "id": image["id"],
                "score": image["score"],
                # "path": image["path"],
                "base64": image_to_base64(cv_img),
                "width": width,
                "height": height,
            }
            processed_images.append(img_data)
            # logger.info(f"Processed image from paper: {image['id']}")
        except Exception as e:
            logger.error(f"Error processing image {image['id']}: {e}")
            logger.error(traceback.format_exc())

    return processed_images