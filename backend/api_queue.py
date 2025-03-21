#pylint: disable=all
from typing import Optional
import os
import json
import pickle
import hashlib
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from robyn import Request
import logging
import traceback

logger = logging.getLogger(__name__)

from api_query_handler import handle_formdata_save
from RAGembedder.multi_modal_embedder import MultimodalEmbedder
from api_dirty import EnumEncoder


type Hash = str;

class QueueStatus(Enum):
    NOT_REGISTERED = 0
    REGISTERED = 1
    FAISS = 2
    RAG = 3
    FINISHED = 4

@dataclass
class QueryHandler:
    """Handles Queries made to backend.

    -> Register query
    -->

    Create hash of specific query and saves query content to
        `./query` folder as an default.

    Spawns a thread to "job queue"

    Returns:
        _type_: _description_
    """
    # Queue -list that rag_thread handles
    faiss_queue: set[int] = field(default_factory=set)
    rag_queue: set[int] = field(default_factory=set)
    # Define where to save everything
    query_path:  os.PathLike = field(default="query")
    upload_path: os.PathLike = field(default="uploads")
    result_path: os.PathLike = field(default="faiss_results")
    embedder:MultimodalEmbedder = field(default=None)
    embedder_thread:threading.Thread = field(default=None)

    def __post_init__(self):
        os.makedirs(self.query_path, exist_ok=True)
        os.makedirs(self.upload_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

        self.faiss_queue = set([])
        self.rag_queue = set([])
        self.embedder = MultimodalEmbedder()
        self.embedder.load_indices()

        def operator(embedder: MultimodalEmbedder):
            """FAISS thread-operator"""
            while True:
                try:
                    # Checking if not empty and ...
                    if self.faiss_queue and (hash := self.faiss_queue.pop()):
                        query = self.get_query_json(hash)
                        result = embedder.search({"text": query["text"]})
                        self.register_faiss_result(hash, result)
                        self.rag_queue.add(hash)
                    else:
                        time.sleep(5)
                except Exception as e:
                    logger.error(f"FAISS THREAD ERROR: {e}")
                    logger.error(traceback.format_exc())

        self.embedder_thread = threading.Thread(target=operator, args=[self.embedder])
        self.embedder_thread.start()


    def register_query(self, request: Request) -> Optional[Hash]:
        hashable_body = pickle.dumps(request.body)
        hash = hashlib.sha256(hashable_body).hexdigest()

        # Check if already registered
        if self.is_registered(hash):
            logger.info(f"Query with hash {hash} already registered.")
            return hash

        try:
            os.makedirs(self.query_path, exist_ok=True)

            query_information = {
                "id": hash,
                "text": request.form_data["query"],
                "uploads": list(request.files.keys()),
                "status" : QueueStatus.REGISTERED
            }

            filepath = os.path.join(self.query_path, hash + ".json")
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(query_information, file, ensure_ascii=False, indent=2, cls=EnumEncoder)

            handle_formdata_save(request.files, self.upload_path)
            self.faiss_queue_push(hash)

        except Exception as e:
            logger.error(f"handle_query_log {e}")
            return None

        logger.info(f"Queue: {list(self.faiss_queue)}")
        return hash

    def faiss_queue_push(self, hash: Hash):
        """Push query and make faiss indices based search"""
        logger.info(f"Queue pushed hash {str(hash)}")
        self.faiss_queue.add(hash)

    # def search_faiss(self, hash: Hash, **kwargs):
    #     query_json = self.get_query_json(hash)
    #     self.embedder.search(query="text")

    def register_faiss_result(self, hash: Hash, result):
        self._change_status(hash, QueueStatus.FAISS)
        self.save_result_json(hash, result)
        # self.faiss_queue.remove(hash)

    def get_faiss_results(self, hash: Hash) -> dict:
        return self.get_result_json(hash) or {"hash": hash, "result": []}

    def query_content(self, hash: Hash):
        pass

    def is_registered(self, hash: Hash) -> bool:
        return self.get_status(hash) != QueueStatus.NOT_REGISTERED

    def get_status(self, hash: Hash) -> QueueStatus:
        # May crash intentionally
        hash_json = self.get_query_json(hash)
        if hash_json:
            return hash_json.get("status", QueueStatus.NOT_REGISTERED)
        else:
            return QueueStatus.NOT_REGISTERED

    def save_result_json(self, hash, results):
        results_json = {"hash": str(hash), "results": results}
        self._json_dump(self.path_to_results_json(hash), results_json)


    def get_query_json(self, hash) -> Optional[dict]:
        """Get sended query json"""
        return self._json_load(self.path_to_query_json(hash=hash))

    def get_result_json(self, hash) -> Optional[dict]:
        """Get faiss results json"""
        return self._json_load(self.path_to_results_json(hash=hash))


    def path_to_results_json(self, hash) -> os.PathLike:
        return os.path.join(self.result_path, str(hash) + ".json")

    def path_to_query_json(self, hash) -> os.PathLike:
        return os.path.join(self.query_path, str(hash) + ".json")

    ################## INTERNAL ##########################

    def _change_status(self, hash: Hash, status: QueueStatus):
        query = self.get_query_json(hash=hash)
        query["status"] = status
        self._json_dump(self.path_to_query_json(hash), query)

    def __spawn_rag_thread(self):
        logger.info("RAG Thread created.")
        pass


    def _json_dump(self, filepath, content):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(content, file, ensure_ascii=False, indent=2, cls=EnumEncoder)
            return True
        except Exception as e:
            logger.error(f"_json_dump {e}")
            return False

    def _json_load(self, filepath) -> Optional[any]:
        json_file = None
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                json_file = json.load(file)
        return json_file




if __name__=="__main__":
    query_handler = QueryHandler()
