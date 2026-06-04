from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.database import get_db, check_db_connection
from app.schemas.prediction import StudentInput, PredictionResponse
from app.services.prediction_service import create_prediction

router = APIRouter(tags=["Predictions"])


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado general del servicio")
    models_loaded: list[str] = Field(..., description="Modelos ML cargados en memoria")
    database: str = Field(..., description="Estado de la conexión a la base de datos ('connected' | 'error')")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    response_description="Resultado del ensemble: nivel de riesgo, tipo de anomalía, features derivadas e intervención recomendada.",
    summary="Detectar anomalía académica",
    description=(
        "Recibe las 26 respuestas Q crudas de la encuesta del semillero, "
        "calcula las 10 features derivadas internamente, ejecuta el ensemble "
        "de 4 modelos y retorna nivel de riesgo, tipo de anomalía e "
        "intervenciones recomendadas."
    ),
)
async def predict(
    request: Request,
    form_data: StudentInput,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    return create_prediction(form_data, db, request.app.state.models)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verifica que los modelos estén cargados y la DB conectada.",
)
async def health(request: Request) -> HealthResponse:
    models = request.app.state.models
    loaded = [k for k in models if not k.endswith("_threshold")]
    db_ok = check_db_connection()
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        database="connected" if db_ok else "error",
    )
