from fastapi import APIRouter

from . import dataset, login, users, ds_ws, data_story

api_router = APIRouter()
api_router.include_router(login.router, prefix="/login", tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(ds_ws.router, prefix="/datascientist_ws", tags=["datasci_ws"])     # Websocket for chatbot
api_router.include_router(dataset.router, prefix="/dataset", tags=["dataset"])
api_router.include_router(data_story.router, prefix="/data_story", tags=["data_story"])

@api_router.get("/")
async def root():
    return {"message": "Backend API for FARM-docker operational !"}
