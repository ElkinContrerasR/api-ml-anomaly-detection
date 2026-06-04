from typing import Literal, Optional
from pydantic import BaseModel, Field


_EXAMPLE_INPUT = {
    "Q1": 1, "Q2": 22, "Q3": 5, "Q4": 4, "Q5": 4,
    "Q6": 4, "Q7": 2, "Q8": 3, "Q9": 7,
    "Q12": 1, "Q13": 2, "Q15": 3, "Q16": 2,
    "Q17": 3, "Q18": 2.0, "Q19": 3, "Q20": 5,
    "Q21": 1, "Q23": 3, "Q24": 3, "Q28": 5,
    "Q29": 17, "Q32": 4, "Q33": 4, "Q34": 3, "Q35": 4,
}

_EXAMPLE_RESPONSE = {
    "prediction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "anomaly_score": 0.72,
    "risk_level": "HIGH",
    "anomaly_type": "Academic",
    "consensus_count": 2,
    "models_flagged": ["ocsvm", "lof"],
    "intervention": {
        "actions": ["Tutoring program", "Study skills workshop"],
        "timeline": "1-2 weeks",
        "owner": "Learning Advisor",
        "frequency": "Bi-weekly check-ins",
        "metrics": ["Assignment completion", "Grade trajectory"],
    },
    "derived_features": {
        "F_Average_Performance": 3.67,
        "F_Academic_Load": 12.0,
        "F_Life_Balance": 0.38,
        "F_Psychological_Stress": 2.5,
        "F_Family_Support": 3.0,
        "F_Grade_Consistency": 0.58,
        "F_Responsibility_Result_Index": 1.06,
        "F_Parental_Education": 5.0,
        "F_Socioeconomic_Risk": 3.0,
        "F_Interest_Performance_Gap": 2.78,
    },
}


class StudentInput(BaseModel):
    # Gender / demographic
    Q1: int = Field(..., ge=0, le=1, description="Género (0=F, 1=M)")
    Q2: int = Field(..., ge=16, le=35, description="Edad")
    # Academic commitment
    Q3: int = Field(..., ge=1, le=5, description="Responsabilidad académica")
    Q4: int = Field(..., ge=1, le=5, description="Aptitud académica autopercibida")
    Q5: int = Field(..., ge=1, le=5, description="Inteligencia autopercibida")
    Q6: int = Field(..., ge=1, le=5, description="Interés en la carrera")
    # Academic load
    Q7: int = Field(..., ge=1, le=8, description="Horas de clase semanales")
    Q8: int = Field(..., ge=0, le=8, description="Horas de trabajo/práctica")
    Q9: int = Field(..., ge=0, le=15, description="Horas de estudio autónomo")
    # Family/support
    Q12: int = Field(..., ge=0, le=7, description="Apoyo familiar académico")
    Q13: int = Field(..., ge=0, le=7, description="Apoyo familiar emocional")
    # Parental education
    Q15: int = Field(..., ge=1, le=6, description="Nivel educativo del padre")
    Q16: int = Field(..., ge=1, le=6, description="Nivel educativo de la madre")
    # Psychological
    Q17: int = Field(..., ge=1, le=5, description="Estrés académico")
    Q18: float = Field(..., ge=1.0, le=4.0, description="Ansiedad")
    Q19: int = Field(..., ge=1, le=5, description="Interés en la carrera (motivación)")
    # Life balance
    Q20: int = Field(..., ge=0, le=10, description="Horas de ocio/recreación")
    Q21: int = Field(..., ge=0, le=1, description="Trabaja actualmente (0=No, 1=Sí)")
    # Psychological wellbeing
    Q23: int = Field(..., ge=1, le=5, description="Empatía")
    Q24: int = Field(..., ge=1, le=5, description="Manejo de conflictos")
    # Trajectory
    Q28: int = Field(..., ge=1, le=10, description="Semestre actual")
    Q29: int = Field(..., ge=15, le=25, description="Edad de ingreso a la universidad")
    # Socioeconomic
    Q32: int = Field(..., ge=1, le=5, description="Estrato socioeconómico")
    # Academic performance
    Q33: int = Field(..., ge=1, le=5, description="Calificación materia 1")
    Q34: int = Field(..., ge=1, le=5, description="Calificación materia 2")
    Q35: int = Field(..., ge=1, le=5, description="Calificación materia 3")

    model_config = {
        "json_schema_extra": {"examples": [_EXAMPLE_INPUT]}
    }


class DerivedFeatures(BaseModel):
    F_Average_Performance: float = Field(..., description="Promedio de calificaciones: (Q33+Q34+Q35)/3")
    F_Academic_Load: float = Field(..., description="Carga académica total: Q7+Q8+Q9 (horas semanales)")
    F_Life_Balance: float = Field(..., description="Balance vida-carga: horas de ocio / (carga+1)")
    F_Psychological_Stress: float = Field(..., description="Estrés psicológico promedio: (Q17+Q18)/2")
    F_Family_Support: float = Field(..., description="Apoyo familiar total: Q12+Q13")
    F_Grade_Consistency: float = Field(..., description="Consistencia de notas: desviación estándar muestral de Q33, Q34, Q35")
    F_Responsibility_Result_Index: float = Field(..., description="Índice responsabilidad/resultado: Q3 / (F_Average_Performance+1)")
    F_Parental_Education: float = Field(..., description="Nivel educativo parental combinado: Q15+Q16")
    F_Socioeconomic_Risk: float = Field(..., description="Riesgo socioeconómico: (6−estrato) + (1−trabaja)")
    F_Interest_Performance_Gap: float = Field(..., description="Brecha interés-desempeño: |Q6 − F_Average_Performance/3|")


class InterventionDetail(BaseModel):
    actions: list[str] = Field(..., description="Lista de acciones de intervención recomendadas")
    timeline: str = Field(..., description="Plazo sugerido para iniciar las acciones")
    owner: str = Field(..., description="Responsable(s) de ejecutar la intervención")
    frequency: str = Field(..., description="Frecuencia de seguimiento")
    metrics: list[str] = Field(..., description="Indicadores para medir el progreso de la intervención")


class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., description="UUID único de la predicción guardada en base de datos")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Puntuación de anomalía ponderada del ensemble [0–1]")
    risk_level: Literal["NORMAL", "MODERATE", "HIGH", "CRITICAL"] = Field(
        ..., description="Nivel de riesgo clasificado según el score del ensemble"
    )
    anomaly_type: Optional[Literal["Academic", "Balance", "Psychological", "Unusual"]] = Field(
        None, description="Categoría de anomalía detectada; null si risk_level es NORMAL"
    )
    consensus_count: int = Field(..., ge=0, le=4, description="Número de modelos que marcaron anomalía (0–4)")
    models_flagged: list[str] = Field(..., description="Nombres de los modelos que marcaron anomalía")
    intervention: Optional[InterventionDetail] = Field(
        None, description="Plan de intervención recomendado; null si risk_level es NORMAL"
    )
    derived_features: DerivedFeatures = Field(
        ..., description="Las 10 features derivadas calculadas internamente antes de pasar al ensemble"
    )

    model_config = {
        "json_schema_extra": {"examples": [_EXAMPLE_RESPONSE]}
    }
