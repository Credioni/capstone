#pylint: disable=all
import os
import json
import logging
from enum import Enum
logger = logging.getLogger(__name__)

class QueueStatus(Enum):
    NOT_IN_SYSTEM = 0
    REGISTERED = 1
    WORKING = 2
    INQUEUE = 3
    FINISHED = 4


def save_query_content():
    pass

def handle_formdata_save(files, folder_path="uploads"):
    """Handle saving files to `folder`.

    Args:
        request (_type_): _description_
        folder (str, optional): _description_. Defaults to "uploads".
    """
    os.makedirs(folder_path, exist_ok=True)

    filenames = []
    for filename, file in files.items():
        filenames.append(filename)
        filepath = os.path.join(folder_path, filename)

        try:
            with open(filepath, "wb") as f:
                f.write(file)
        except Exception as e:
            logger.error(f"handle_formdata_save {e}")
    return filenames

def handle_query_log(uuid, form_data, filenames, folder_path="queries"):
    try:
        os.makedirs(folder_path, exist_ok=True)

        uuid = str(uuid)
        query_information = {
            "id": uuid,
            "text": form_data["query"],
            "uploads": filenames,
            "status" : QueueStatus.REGISTERED
        }

        filepath = os.path.join(folder_path, uuid + ".json")
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(query_information, file, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"handle_query_log {e}")
        return False

    return True

