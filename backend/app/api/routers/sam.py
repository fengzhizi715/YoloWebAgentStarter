from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_settings, get_storage
from app.core.config import Settings
from app.core.storage import Storage
from app.sam.schemas import SamPredictRequest, SamPredictResponse
from app.sam.service import SamService

router = APIRouter(prefix="/sam", tags=["sam"])


@router.post("/predict", response_model=SamPredictResponse)
def predict(
    payload: SamPredictRequest,
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> SamPredictResponse:
    return SamService(storage, settings).predict(session, payload)
