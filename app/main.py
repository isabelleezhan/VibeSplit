from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import router as auth_router
from app.clusters import router as clusters_router
from app.db import init_db
from app.playlists import router as playlists_router
from app.tracks import router as tracks_router
from app.writeback import router as writeback_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="VibeSplit", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(playlists_router)
app.include_router(tracks_router)
app.include_router(clusters_router)
app.include_router(writeback_router)

@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, VibeSplit!"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
