import uvicorn
from api.config import get_settings
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from api.views import api_router

def get_app():

    app = FastAPI(
        title=get_settings().PROJECT_NAME,
        docs_url=f"/api/{get_settings().VERSION}/docs",
        redoc_url=f"/api/{get_settings().VERSION}/redoc",
        openapi_url=f"/api/{get_settings().VERSION}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=f"/api/{get_settings().VERSION}")

    return app