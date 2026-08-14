from fastapi import APIRouter

from app.api.routers import datasets, models, sam, settings, system, training
from app.logs.router import router as logs_router

api_router = APIRouter(prefix="/api")
api_router.include_router(system.router)
api_router.include_router(datasets.router)
api_router.include_router(datasets.file_router)
api_router.include_router(training.router)
api_router.include_router(models.router)
api_router.include_router(sam.router)
api_router.include_router(settings.router)
api_router.include_router(logs_router)
