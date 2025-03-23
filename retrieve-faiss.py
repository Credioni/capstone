import os
from huggingface_hub import snapshot_download

# Upload the FAISS index file
dir_path = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.join("backend", "RAGembedder")

local_dir = snapshot_download(
    repo_id="Credioni/capstone-faiss",
    repo_type="dataset",
    local_dir=os.path.join(dir_path, folder_path)
)
