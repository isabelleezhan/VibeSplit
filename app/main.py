from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.auth import router as auth_router
from app.db import init_db
from app.playlists import router as playlists_router
from app.tracks import router as tracks_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="VibeSplit", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(playlists_router)
app.include_router(tracks_router)


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, VibeSplit!"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
