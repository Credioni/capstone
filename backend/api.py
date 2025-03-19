# pylint: disable=all
import os
import logging
import urllib.parse
from RAG import ArXivRAGSystem
####################### ENDPOINT HANDLING ####################
from robyn import Robyn, ALLOW_CORS
app = Robyn(__file__)
####################### API OF APIS ####################
from api_process_images import process_images
from api_saved_queries import contains_response, save_response
from api_paper_retrieving import fetch_arxiv_metadata
####################### CORS ####################
ALLOW_CORS(app, origins = ["http://localhost:3000"])

####################### LOGGING ####################
from logging_formatter import CustomFormatter

logging.basicConfig(
    level=logging.DEBUG,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        CustomFormatter(),
        logging.FileHandler("log/api.log"),
    ]
)

logger = logging.getLogger(__name__)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setFormatter(CustomFormatter())
# logger.addHandler(ch)


####################### RAG ####################
# Configuration
def load_configuration():
    """ Load the configuration for the RAG system.
    Returns:
        A dictionary containing the configuration parameters
    """
    # Get the directory containing this file
    filedir = os.path.dirname(os.path.abspath(__file__))

    config = {
        "faiss_index_path": os.path.join(filedir, "data", "faiss_index", "final_index.index"),
        "mapping_path": os.path.join(filedir, "data", "faiss_index", "final_mapping.json"),
        "projection_path": os.path.join(filedir, "data", "faiss_index", "projection.pt"),
        "image_folder": os.path.join(filedir, "data", "images")
    }

    # Check if all files exist
    for key, path in config.items():
        if not os.path.exists(path):
            logger.warning(f"File at {key} does not exist: {path}")

    return config

# Load configuration
config = load_configuration()
logger.info("Configuration loaded:")
for key, value in config.items():
    logger.info(f"  {key}: {value}")

rag_system = ArXivRAGSystem(config=config)


####################### BACKEND API CALL INTERFACE ####################
@app.get("/")
async def home(request):
    return "Hello, world!"

@app.get("/query")
async def query(request, query_params):
    query_params = query_params.to_dict()['q']
    query_text = " ".join(query_params)
    query_text = urllib.parse.unquote(query_text)

    print(f"Quering<{query_text}>...")

    # Check if query is saved, if so send it
    if (response := contains_response(query_text)) is not None:
        logger.info(f"Query found in saved queries!")
        logger.info(f"Query processed successfully!")
        return response
    else:
        logger.info(f"Query not found in saved queries.")

    # Generate response using RAG
    metadata = rag_system.query(query_text, k=1, score_threshold=0.7)
    answer = metadata['answer']

    # Process images if any and if requested
    images = []
    if metadata['images']:
        images = process_images(metadata['images'])
        logger.info(f"Processed {len(images)} images for response")

    sources = [{
        "paper_id": src["paper_id"],
        "score": src["score"],
        "metadata": fetch_arxiv_metadata(src["paper_id"]),
        # Filter papers with same paper_id
        "images": list(filter(lambda x: x["paper_id"] == src["paper_id"], images)),
    } for src in metadata['sources']]

    logger.info(f"Query processed successfully, answer:\n{answer}")

    # JSON-dict
    response_data = {
        "status" : "success",
        "answer" : answer,
        "sources": sources
    }

    save_response(query_text, response_data)

    return response_data


app.start(port=8080)

def main():
    pass

if __name__ == "__main__":
   main()