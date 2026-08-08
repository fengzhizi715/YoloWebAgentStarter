from fastapi import APIRouter

from app.api.routers import datasets, models, system, training

api_router = APIRouter(prefix="/api")
api_router.include_router(system.router)
api_router.include_router(datasets.router)
api_router.include_router(datasets.file_router)
api_router.include_router(training.router)
api_router.include_router(models.router)
