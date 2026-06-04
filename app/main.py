import os
import pickle
from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html

from app.config import settings
from app.routers import predict


def _load_pkl(store: str, name: str):
    with open(os.path.join(store, name), "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = settings.model_store_path
    app.state.models = {
        "ocsvm": _load_pkl(store, "ocsvm.pkl"),
        "lof": _load_pkl(store, "lof.pkl"),
        "iso_forest": _load_pkl(store, "iso_forest.pkl"),
        "autoencoder": tf.keras.models.load_model(
            os.path.join(store, "autoencoder.keras")
        ),
        "autoencoder_threshold": _load_pkl(store, "autoencoder_threshold.pkl"),
        "scaler": _load_pkl(store, "scaler_features.pkl"),
        "semillero_means": _load_pkl(store, "semillero_means.pkl"),
        "semillero_stds": _load_pkl(store, "semillero_stds.pkl"),
    }
    yield
    app.state.models.clear()


app = FastAPI(
    title="MADS API",
    description=(
        "**MAMBA Anomaly Detection System** — Detección de anomalías académicas "
        "para el semillero MAMBA de CORHUILA.\n\n"
        "Recibe las 26 respuestas Q crudas de la encuesta, calcula internamente "
        "las 10 features derivadas y ejecuta un **ensemble de 4 modelos** "
        "para clasificar el nivel de riesgo y recomendar intervenciones.\n\n"
        "## Modelos del ensemble\n\n"
        "| Modelo | Peso |\n"
        "|---|---|\n"
        "| OC-SVM | 40 % |\n"
        "| LOF | 30 % |\n"
        "| Isolation Forest | 20 % |\n"
        "| Autoencoder | 10 % |\n\n"
        "## Umbrales de riesgo\n\n"
        "| Score | Nivel |\n"
        "|---|---|\n"
        "| ≥ 0.75 | `CRITICAL` |\n"
        "| ≥ 0.50 | `HIGH` |\n"
        "| ≥ 0.25 | `MODERATE` |\n"
        "| < 0.25 | `NORMAL` |"
    ),
    version="1.0.0",
    contact={"name": "Semillero MAMBA", "email": "semillero@corhuila.edu.co"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {
            "name": "Predictions",
            "description": (
                "Endpoints para inferencia del ensemble de detección de anomalías. "
                "Recibe los datos crudos del estudiante, ejecuta el pipeline completo "
                "(features → normalización → ensemble) y retorna el nivel de riesgo "
                "con el plan de intervención correspondiente."
            ),
        }
    ],
    lifespan=lifespan,
    redoc_url=None,
)

@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title="API Documentation",
        redoc_js_url="https://unpkg.com/redoc@2.1.3/bundles/redoc.standalone.js",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
