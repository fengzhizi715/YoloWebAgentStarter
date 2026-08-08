from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_storage
from app.core.storage import Storage
from app.models.schemas import ModelVersionList, ModelVersionResponse, ModelVersionUpdate
from app.models.service import ModelService


router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelVersionList)
def list_models(
    dataset_id: str | None = Query(default=None),
    include_archived: bool = Query(default=False),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ModelVersionList:
    return ModelService(storage).list_models(session, dataset_id, include_archived)


@router.get("/{model_id}", response_model=ModelVersionResponse)
def get_model(
    model_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ModelVersionResponse:
    return ModelService(storage).get_model_response(session, model_id)


@router.patch("/{model_id}", response_model=ModelVersionResponse)
def update_model(
    model_id: str,
    payload: ModelVersionUpdate,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ModelVersionResponse:
    return ModelService(storage).update_model(session, model_id, payload)


@router.post("/{model_id}/archive", response_model=ModelVersionResponse)
def archive_model(
    model_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ModelVersionResponse:
    return ModelService(storage).archive_model(session, model_id)


@router.post("/{model_id}/restore", response_model=ModelVersionResponse)
def restore_model(
    model_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ModelVersionResponse:
    return ModelService(storage).restore_model(session, model_id)


@router.delete("/{model_id}", status_code=204, response_class=Response, response_model=None)
def delete_model(
    model_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> None:
    ModelService(storage).delete_model(session, model_id)


@router.get("/{model_id}/download")
def download_model(
    model_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> FileResponse:
    path = ModelService(storage).download_path(session, model_id)
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.post("/{model_id}/export-onnx", response_model=ModelVersionResponse)
def export_onnx(
    model_id: str,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> ModelVersionResponse:
    return ModelService(storage).export_onnx(session, model_id)
