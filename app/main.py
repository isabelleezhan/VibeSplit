from fastapi import FastAPI

app = FastAPI(title="VibeSplit")


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, VibeSplit!"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
