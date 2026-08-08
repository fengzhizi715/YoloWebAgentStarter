from fastapi import APIRouter

from app.api.routers import datasets, system

api_router = APIRouter(prefix="/api")
api_router.include_router(system.router)
api_router.include_router(datasets.router)
api_router.include_router(datasets.file_router)
