#pylint: disable=all
import arxiv

def get_arxiv_metadata(arxiv_id):
    pass

def fetch_arxiv_metadata(arxiv_id):
    """Fetches title, authors, and abstract from arXiv using the arxiv_id."""
    try:
        search = arxiv.Search(id_list=[arxiv_id])
    except Exception as e:
        search = None

    metadata = {
        "paper_id": arxiv_id,
        "title": "unknown",
        "authors": "unknown",
        "abstract": "unknown",
        "doi": "unknown",
        "published": "unknown",
        "url": "unknown"
    }

    if search is not None and (paper := next(search.results(), None)) is not None:
        metadata = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "doi": paper.doi if paper.doi else "DOI Not Available",
            "published": paper.published.strftime("%Y-%m-%d %H:%M:%S"),
            "url": paper.entry_id
        }
        return metadata
    else:
        return metadata

