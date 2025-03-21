#pylint: disable=all
import os
import arxiv
import json
from typing import Optional
from dataclasses import dataclass, field
import logging
logger = logging.getLogger(__name__)


@dataclass
class ArxivMetadataHandler:
    """ArXiv metadata handler
        `Fetched`, `Saves` and `Retrieves` paper information to `FrontEnd`.
    """
    folder_path: os.path = field(default=os.path.join("arxiv_metadata"))

    def get_paper_information(self, paper_id):
        if (metadata := self.load_metadata(paper_id=paper_id)) is None:
            metadata = self.__fetch_arxiv_metadata(paper_id=paper_id)

        paper_json = {
            "paper_id": paper_id,
            "title": metadata["title"],
            "authors": [author for author in metadata["authors"]],
            "abstract": metadata["abstract"],
            "doi": metadata["doi"] if metadata["doi"] else "DOI Not Available",
            "published": metadata["published"],
            "url": metadata["url"]
        }
        return paper_json

    def __fetch_arxiv_metadata(self, paper_id) -> dict:
        """Fetches title, authors, and abstract from arXiv using the arxiv_id."""
        try:
            search  = arxiv.Search(id_list=[paper_id])
            results = arxiv.Client().results(search)
        except Exception as e:
            logger.error(e)
            results = iter([])

        if (result := next(results, None)) is not None:
            metadata = {
                "paper_id": paper_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "doi": result.doi if result.doi else "DOI Not Available",
                "published": result.published.strftime("%Y-%m-%d %H:%M:%S"),
                "url": result.pdf_url,
            }
            return self.save_metadata(paper_id=paper_id, metadata=metadata)
        else:
            return self.save_metadata(paper_id=paper_id, metadata=None)

    def save_metadata(self, paper_id, metadata) -> dict:
        if not metadata:
            metadata = {
                "paper_id": paper_id,
                "title": "",
                "authors": [],
                "abstract": "",
                "doi": "",
                "published": "",
                "url": ""
            }

        paper_path = self._paper_path(paper_id=paper_id)
        self._json_dump(paper_path, metadata)

        return metadata

    def load_metadata(self, paper_id) -> Optional[dict]:
        try:
            metadata = self._paper_metadata_json(paper_id=paper_id)
            return metadata if metadata else None
        except Exception as e:
            logger.error(f"load_metadata {e}")
            return None

    def _paper_metadata_json(self, paper_id: str) -> Optional[dict]:
        if metadata := self._read_paper(paper_id):
            return metadata
        else:
            return None

    def _read_paper(self, paper_id: str) -> Optional[dict]:
        return self._json_load(self._paper_path(paper_id=paper_id))

    def _paper_path(self, paper_id: str) -> os.PathLike:
        return os.path.join(self.folder_path, paper_id + ".json")

    def _json_dump(self, filepath, content):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(content, file, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"_json_dump {e}")
            return False

    def _json_load(self, filepath):
        json_file = None
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                json_file = json.load(file)
        return json_file

def test_arxiv():
    arxiv_handler = ArxivMetadataHandler()
    metadata = arxiv_handler.get_paper_information("0704.0001")
    assert metadata is not None
    return metadata

if __name__=="__main__":
    print(*test_arxiv().items(), sep="\n")
