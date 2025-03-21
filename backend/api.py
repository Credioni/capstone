# pylint: disable=all
import os
# from RAG import ArXivRAGSystem
####################### ENDPOINT HANDLING ####################
from robyn import Robyn, ALLOW_CORS
from robyn.robyn import Request, QueryParams




app = Robyn(__file__)
####################### API OF APIS ####################
from api_queue import QueryHandler
from api_dirty import init_logger
from api_process_images import process_images
from api_saved_queries import contains_response, save_response
from api_query_handler import handle_formdata_save, handle_query_log
####################### CORS ####################
ALLOW_CORS(app, origins = ["http://localhost:3000"])
####################### LOGGING ####################
logger = init_logger()
####################### CONFIG AND RAG ####################
# Handles the queue of queries
# config = load_configuration(logger)
# rag_system = ArXivRAGSystem(config=config)
####################### BACKEND API CALL INTERFACE ####################

QUERY_HANDLER = QueryHandler()


@app.get("/")
async def home(request):
    return "Hello, world!"



@app.post("/query")
async def query(request: Request, query_params):
    query_hash = QUERY_HANDLER.register_query(request=request)
    return {
        "status": "success",
        "query_id": query_hash,
    }



@app.get("/result")
async def result_status(request: Request, query_params: QueryParams):
    print(query_params)
    if (hash := query_params.get("q")) is not None:
        return QUERY_HANDLER.get_result_json(hash) or {}
    else:
        return { "status": "waiting" }



app.start(port=8080)
