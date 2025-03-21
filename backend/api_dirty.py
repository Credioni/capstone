#pylint: disable=all
import os
import json
from enum import Enum

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # Return the value of the Enum
        return super().default(obj)

# # Configuration
# def load_configuration(logger):
#     """ Load the configuration for the RAG system.
#     Returns:
#         A dictionary containing the configuration parameters
#     """
#     # Get the directory containing this file
#     filedir = os.path.dirname(os.path.abspath(__file__))

#     config = {
#         "faiss_index_path": os.path.join(filedir, "data", "faiss_index", "final_index.index"),
#         "mapping_path": os.path.join(filedir, "data", "faiss_index", "final_mapping.json"),
#         "projection_path": os.path.join(filedir, "data", "faiss_index", "projection.pt"),
#         "image_folder": os.path.join(filedir, "data", "images")
#     }

#     # Check if all files exist
#     for key, path in config.items():
#         if not os.path.exists(path):
#             logger.warning(f"File at {key} does not exist: {path}")

#     # Log configs
#     for key, value in config.items():
#         logger.info(f"  {key}: {value}")

#     return config


import logging
from logging_formatter import CustomFormatter

def init_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            CustomFormatter(),
            # logging.FileHandler("log/api.log"),
        ]
    )

    return logging.getLogger(__name__)