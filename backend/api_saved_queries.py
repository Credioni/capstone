#pylint: disable=all
import os
import json
import logging
import hashlib

logger = logging.getLogger(__name__)

PATH_SAVED_QUERIES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "saved_queries"
)

def query2filepath(query):
    """Convert query to hashings"""
    filename = f"{query}.json"
    filepath = os.path.join(PATH_SAVED_QUERIES, filename)
    return filepath

def contains_response(query):
    """Check if a response exists for the given query."""
    filepath = query2filepath(query)
    print("", filepath)
    print("", os.path.isfile(filepath))
    try:
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                response = json.load(file)
            return response
        else:
            return None
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading response file for query: {e}")
        return None

def save_response(query, response):
    """Save a response for the given query."""
    filepath = query2filepath(query)

    # Ensure directory exists
    os.makedirs(PATH_SAVED_QUERIES, exist_ok=True)

    logger.info(f"Saving response to {filepath}")

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(response, file, ensure_ascii=False, indent=2)
        return True
    except (IOError, TypeError) as e:
        logger.error(f"Error saving response for query: {e}")
        return False

