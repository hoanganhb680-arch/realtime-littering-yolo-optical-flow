# ==============================================================================
# ENTRY POINT
# ==============================================================================
import asyncio
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from Config import settings
from violation_router import router as violation_router
from stream_router    import router as stream_router, _queue_broadcaster, start_detector_thread
import db as _db

app = FastAPI(title=settings.APP_TITLE)

# --- CORS: cho phép Vite dev server (port 5173) và production ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(violation_router)
app.include_router(stream_router)


# --- Startup / Shutdown ---
@app.on_event("startup")
async def on_startup() -> None:
    _db.init_db()
    # Chạy WebSocket broadcaster ngầm trong event loop
    asyncio.create_task(_queue_broadcaster())
    # Chạy detector trong background thread (non-blocking)
    start_detector_thread()


if __name__ == "__main__":
    uvicorn.run("server:app", host=settings.HOST, port=settings.PORT, reload=False)