import uvicorn
from fastapi import FastAPI
from routers.v1 import chat

app = FastAPI(title="Chat API")

# El router ya tiene prefix="/api/v1/messages"
app.include_router(chat.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
