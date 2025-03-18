# pylint: disable=all
import os
import urllib.parse
from RAG import ArXivRAGSystem


from robyn import Robyn, ALLOW_CORS
app = Robyn(__file__)
ALLOW_CORS(app, origins = ["http://localhost:3000"])

####################### RAG ####################
# Configuration
filedir = os.path.dirname(os.path.abspath(__file__))
# print(f"{filedir = }")

config = {
    "faiss_index_path": os.path.join(filedir, "data", "faiss_index", "final_index.index"),
    "mapping_path": os.path.join(filedir, "data", "faiss_index", "final_mapping.json"),
    "projection_path": os.path.join(filedir, "data", "faiss_index", "projection.pt"),
    "image_folder": os.path.join(filedir, "data", "images")
}

print(*config.items(), sep="\n")

rag_system = ArXivRAGSystem(config=config)


####################### BACKEND API CALL INTERFACE ####################
@app.get("/")
async def h(request):
    return "Hello, world!"

@app.get("/query")
async def query(request, query_params):
    query_params = query_params.to_dict()['q']
    query_text = " ".join(query_params)
    query_text = urllib.parse.unquote(query_text)

    print(f"Quering<{query_text}>...")

    answer = rag_system.query(query_text, k=5, score_threshold=0.7)['answer']
    print("################# ANSWER #################")
    print(answer)
    print("################# END #################")
    return {"message": "Hello, world!", "answer": str(answer)}


app.start(port=8080)

def main():
    pass

if __name__ == "__main__":
   main()