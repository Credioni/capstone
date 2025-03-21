#pylint: disable=all
import json
from multi_modal_embedder import MultimodalEmbedder

def zero_shot(query, **kwargs):
    embedder = MultimodalEmbedder()
    embedder.load_indices()

    result = embedder.search(query=query, **kwargs)

    # Pretty print
    json_formatted_str = json.dumps(result, indent=2)
    print(json_formatted_str)


if __name__=="__main__":
    zero_shot(query="Quantum Mechanics")
