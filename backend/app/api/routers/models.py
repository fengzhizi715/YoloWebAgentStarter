from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_storage
from app.core.storage import Storage
from app.models.schemas import InferenceResult, ModelCompareRequest, ModelCompareResponse, ModelEvaluationRecordResponse, ModelEvaluationRequest, ModelTestRecordResponse, ModelVersionList, ModelVersionResponse, ModelVersionUpdate, PreAnnotationRequest, PreAnnotationResponse
from app.models.service import ModelService


router = APIRouter(prefix="/models", tags=["models"])


@router.post("/compare", response_model=ModelCompareResponse)
def compare_models(payload: ModelCompareRequest, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> ModelCompareResponse:
    return ModelCompareResponse.model_validate(ModelService(storage).compare(session, payload.baseline_model_id, payload.candidate_model_id))


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


@router.post("/{model_id}/test", response_model=InferenceResult)
async def test_model(
    model_id: str,
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0, le=1),
    iou: float = Query(default=0.45, ge=0, le=1),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> InferenceResult:
    image_bytes = await file.read()
    service = ModelService(storage)
    result = service.run_test_inference(session, model_id, image_bytes, confidence, iou)
    service.save_test_record(session, model_id, file.filename or "test.jpg", image_bytes, result)
    return InferenceResult.model_validate(result)


@router.get("/{model_id}/tests", response_model=list[ModelTestRecordResponse])
def list_model_tests(model_id: str, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> list[ModelTestRecordResponse]:
    return [ModelTestRecordResponse.model_validate(item) for item in ModelService(storage).list_test_records(session, model_id)]


@router.post("/{model_id}/evaluate", response_model=ModelEvaluationRecordResponse, status_code=201)
def evaluate_model(model_id: str, payload: ModelEvaluationRequest, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> ModelEvaluationRecordResponse:
    return ModelEvaluationRecordResponse.model_validate(ModelService(storage).evaluate(session, model_id, payload.split, payload.confidence, payload.iou))


@router.get("/{model_id}/evaluations", response_model=list[ModelEvaluationRecordResponse])
def list_model_evaluations(model_id: str, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> list[ModelEvaluationRecordResponse]:
    return [ModelEvaluationRecordResponse.model_validate(item) for item in ModelService(storage).list_evaluations(session, model_id)]


@router.post("/{model_id}/preannotate", response_model=PreAnnotationResponse)
def preannotate_images(model_id: str, payload: PreAnnotationRequest, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> PreAnnotationResponse:
    return PreAnnotationResponse(model_id=model_id, dataset_id=payload.dataset_id, images=ModelService(storage).preannotate(session, model_id, payload.dataset_id, payload.image_ids, payload.confidence, payload.iou))
