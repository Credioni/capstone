###
import base64
import io

def image_to_base64(img):
    """
    Convert a PIL Image to a base64-encoded string.

    Args:
        img: PIL Image object

    Returns:
        Base64-encoded string representation of the image
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def process_images(images, logger):
    """
    Process image data for API response.

    Args:
        images: List of image objects from RAG system

    Returns:
        List of processed image data with base64 encoding
    """
    processed_images = []

    for img_info in images:
        try:
            img_data = {
                "paper_id": img_info["paper_id"],
                "path": img_info["path"],
                "base64": image_to_base64(img_info["image"]),
                "width": img_info["image"].width,
                "height": img_info["image"].height
            }
            processed_images.append(img_data)
            logger.info(f"Processed image from paper: {img_info['paper_id']}")
        except Exception as e:
            logger.error(f"Error processing image {img_info.get('path', 'unknown')}: {e}")

    return processed_images