from fastapi import FastAPI
from .api.routes import router
import os

app = FastAPI(
    title="Swafinix AI Admission Inquiry Assistant",
    description="AI-powered assistant to answer admission queries based on pre-indexed policy documents.",
    version="1.0.0",
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Swafinix AI Admission Inquiry Assistant API!"}
