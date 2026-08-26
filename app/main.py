from fastapi import FastAPI

from app.auth import router as auth_router
from app.playlists import router as playlists_router

app = FastAPI(title="VibeSplit")
app.include_router(auth_router)
app.include_router(playlists_router)


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, VibeSplit!"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
