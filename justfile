# use PowerShell instead of sh:
set shell := ["powershell.exe", "-c"]



setup-all:
    just install-front
    just install-backend
    just retrieve-faiss

install-front:
    cd frontend | pnpm install
    cd frontend | pnpm build

install-backend:
    py -3.12 -m pip install -r requirements.txt

retrieve-faiss:
    py -3.12 .\retrieve-faiss.py

##############################

# Resets and builds the embeddings except arxiv paper
build-faiss:
    cd backend/rag | py -3.12 .\build.py

# Zero-shot to test that embeddings work
zero-shot:
    cd backend/rag | py -3.12 .\zero_shot.py

# Run backend with dev mode
run-back:
    cd backend | py -3.12 .\api.py

dev-back:
    cd backend | py -3.12 -m robyn .\api.py --dev

run-front:
    cd frontend | pnpm install
    cd frontend | pnpm build
    just dev-front

# Run frontend with dev mode
dev-front:
    cd frontend | pnpm run dev


# Get line count of all subdirectories, including .git-folder.
linecount type:
    dir . -filter "*{{type}}" -Recurse -name | foreach{(GC $_).Count} | measure-object -sum
