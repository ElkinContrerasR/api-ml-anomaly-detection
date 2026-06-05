# MADS API — MAMBA Anomaly Detection System

API REST de detección de anomalías académicas desarrollada como proyecto de grado para el
**Semillero MAMBA** de la Corporación Universitaria del Huila (**CORHUILA**).

Recibe las respuestas crudas de una encuesta aplicada a estudiantes, calcula internamente
un conjunto de features derivadas y ejecuta un **ensemble de 4 modelos de aprendizaje no
supervisado** para clasificar el nivel de riesgo académico y emitir un plan de intervención
personalizado.

---

## Tabla de contenidos

- [Contexto del problema](#contexto-del-problema)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Pipeline de ML](#pipeline-de-ml)
- [Endpoints de la API](#endpoints-de-la-api)
- [Variables de entrada (26 Q)](#variables-de-entrada-26-q)
- [Features derivadas (10 F)](#features-derivadas-10-f)
- [Niveles de riesgo e intervenciones](#niveles-de-riesgo-e-intervenciones)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Variables de entorno](#variables-de-entorno)
- [Casos de prueba](#casos-de-prueba)
- [Stack tecnológico](#stack-tecnológico)

---

## Contexto del problema

El semillero MAMBA recopila datos académicos, socioeconómicos y psicológicos de sus
estudiantes a través de una encuesta estandarizada. El objetivo del sistema es detectar de
forma temprana perfiles atípicos que puedan indicar riesgo de deserción o bajo rendimiento,
para que los directivos y asesores puedan actuar antes de que el problema se agrave.

La detección se realiza mediante técnicas de **anomaly detection** (no existe una etiqueta
"en riesgo / normal" en los datos históricos), combinando cuatro algoritmos con distintas
sensibilidades para maximizar la cobertura.

---

## Arquitectura del sistema

```
Cliente  ──POST /predict──►  FastAPI
                               │
                        app/ml/pipeline.py
                        Calcula 10 F_ derivadas
                        Normaliza con MinMaxScaler
                               │
                        app/ml/ensemble.py
                        ┌──────────────────────┐
                        │  OC-SVM      (40 %)  │
                        │  LOF         (30 %)  │
                        │  Iso. Forest (20 %)  │
                        │  Autoencoder (10 %)  │
                        └──────────────────────┘
                        Score ponderado [0, 1]
                               │
                        app/ml/classifier.py
                        Nivel de riesgo + tipo
                        Plan de intervención
                               │
                        PostgreSQL (trazabilidad)
                               │
                        ◄── JSON response ──────
```

---

## Pipeline de ML

### 1. Entrenamiento (`train_models.py`)

El script lee el dataset crudo `data/RespuestasSemillero_completo.json` (81 registros del
semillero), calcula las 10 features derivadas y genera 8 artefactos en `models_store/`:

| Artefacto | Descripción |
|---|---|
| `scaler_features.pkl` | MinMaxScaler ajustado sobre los rangos reales del semillero |
| `semillero_means.pkl` | Media de cada variable Q (usada para z-scores) |
| `semillero_stds.pkl` | Desviación estándar muestral de cada Q |
| `ocsvm.pkl` | One-Class SVM entrenado |
| `lof.pkl` | Local Outlier Factor (novelty=True) |
| `iso_forest.pkl` | Isolation Forest entrenado |
| `autoencoder.keras` | Autoencoder Keras (10→32→16→32→10) |
| `autoencoder_threshold.pkl` | Umbral de reconstrucción (percentil 95 del MSE de entrenamiento) |

### 2. Hiperparámetros

```python
OneClassSVM(kernel='rbf', gamma='auto', nu=0.10)

LocalOutlierFactor(n_neighbors=20, contamination=0.10, novelty=True)

IsolationForest(contamination=0.10, n_estimators=100, random_state=42)

# Autoencoder: Dense(32,relu) → Dense(16,relu) → Dense(32,relu) → Dense(10,sigmoid)
# optimizer='adam', loss='mse', epochs=50, batch_size=32, validation_split=0.2
```

### 3. Pesos del ensemble

| Modelo | Peso |
|---|---|
| OC-SVM | 40 % |
| LOF | 30 % |
| Isolation Forest | 20 % |
| Autoencoder | 10 % |

El **anomaly score** es la suma ponderada de los flags de cada modelo (1 = anomalía, 0 = normal).

---

## Endpoints de la API

### `POST /predict`

Recibe las 26 variables Q en JSON y retorna el análisis completo.

**Request:**
```json
{
  "Q1": 1, "Q2": 22, "Q3": 5, "Q4": 4, "Q5": 4,
  "Q6": 4, "Q7": 2, "Q8": 3, "Q9": 7,
  "Q12": 1, "Q13": 2, "Q15": 3, "Q16": 2,
  "Q17": 3, "Q18": 2.0, "Q19": 3, "Q20": 5,
  "Q21": 1, "Q23": 3, "Q24": 3, "Q28": 5,
  "Q29": 17, "Q32": 4, "Q33": 4, "Q34": 3, "Q35": 4
}
```

**Response 200:**
```json
{
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
    "metrics": ["Assignment completion", "Grade trajectory"]
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
    "F_Interest_Performance_Gap": 2.78
  }
}
```

> `anomaly_type` e `intervention` son `null` cuando `risk_level` es `"NORMAL"`.

---

### `GET /health`

Verifica el estado de la API, los modelos cargados y la conexión a la base de datos.

**Response 200:**
```json
{
  "status": "ok",
  "models_loaded": ["ocsvm", "lof", "iso_forest", "autoencoder", "scaler", "means", "stds"],
  "database": "connected"
}
```

---

## Variables de entrada (26 Q)

El cliente envía directamente las respuestas crudas de la encuesta. La API deriva las
features internamente.

| Variable | Descripción | Rango válido |
|---|---|---|
| Q1 | Género | 0 (F) – 1 (M) |
| Q2 | Edad | 16 – 35 |
| Q3 | Responsabilidad académica | 1 – 5 |
| Q4 | Aptitud académica autopercibida | 1 – 5 |
| Q5 | Inteligencia autopercibida | 1 – 5 |
| Q6 | Interés en la carrera | 1 – 5 |
| Q7 | Horas de clase semanales | 1 – 8 |
| Q8 | Horas de trabajo/práctica | 0 – 8 |
| Q9 | Horas de estudio autónomo | 0 – 15 |
| Q12 | Apoyo familiar académico | 0 – 7 |
| Q13 | Apoyo familiar emocional | 0 – 7 |
| Q15 | Nivel educativo del padre | 1 – 6 |
| Q16 | Nivel educativo de la madre | 1 – 6 |
| Q17 | Estrés académico | 1 – 5 |
| Q18 | Ansiedad | 1.0 – 4.0 |
| Q19 | Motivación hacia la carrera | 1 – 5 |
| Q20 | Horas de ocio/recreación | 0 – 10 |
| Q21 | Trabaja actualmente | 0 (No) – 1 (Sí) |
| Q23 | Empatía | 1 – 5 |
| Q24 | Manejo de conflictos | 1 – 5 |
| Q28 | Semestre actual | 1 – 10 |
| Q29 | Edad de ingreso a la universidad | 15 – 25 |
| Q32 | Estrato socioeconómico | 1 – 5 |
| Q33 | Calificación materia 1 | 1 – 5 |
| Q34 | Calificación materia 2 | 1 – 5 |
| Q35 | Calificación materia 3 | 1 – 5 |

> Las variables Q10, Q11, Q14, Q22, Q25–Q27, Q30–Q31 son campos de texto libre o no
> intervienen en ninguna fórmula; no forman parte del input.

---

## Features derivadas (10 F)

Calculadas internamente en [`app/ml/pipeline.py`](app/ml/pipeline.py) a partir de las Q crudas:

| Feature | Fórmula | Significado |
|---|---|---|
| F_Average_Performance | `(Q33 + Q34 + Q35) / 3` | Promedio de calificaciones |
| F_Academic_Load | `Q7 + Q8 + Q9` | Carga horaria académica total |
| F_Life_Balance | `Q20 / (F_Academic_Load + 1)` | Proporción ocio / carga |
| F_Psychological_Stress | `(Q17 + Q18) / 2` | Estrés psicológico promedio |
| F_Family_Support | `Q12 + Q13` | Apoyo familiar total |
| F_Grade_Consistency | `std([Q33, Q34, Q35], ddof=1)` | Variabilidad de notas |
| F_Responsibility_Result_Index | `Q3 / (F_Average_Performance + 1)` | Responsabilidad vs. resultado |
| F_Parental_Education | `Q15 + Q16` | Nivel educativo parental combinado |
| F_Socioeconomic_Risk | `(6 − Q32) + (1 − Q21)` | Riesgo socioeconómico |
| F_Interest_Performance_Gap | `\|Q6 − F_Average_Performance / 3\|` | Brecha interés–desempeño |

Después del cálculo, el vector de 10 features se normaliza con el `scaler_features.pkl`
antes de entrar al ensemble.

---

## Niveles de riesgo e intervenciones

### Umbrales del ensemble score

| Score | Nivel de riesgo |
|---|---|
| ≥ 0.75 | `CRITICAL` |
| ≥ 0.50 | `HIGH` |
| ≥ 0.25 | `MODERATE` |
| < 0.25 | `NORMAL` |

### Tipos de anomalía

La clasificación del tipo se determina calculando z-scores de las Q crudas respecto a las
medias del semillero e identificando el grupo con mayor desviación acumulada:

| Tipo | Variables que lo determinan |
|---|---|
| Academic | Q3, Q6, Q7, Q8, Q9, Q33, Q34, Q35 |
| Balance | Q7, Q20, Q21, Q32, Q12, Q13 |
| Psychological | Q4, Q5, Q17, Q18, Q19, Q23, Q24 |
| Unusual | Q2, Q28, Q29, Q1, Q15, Q16 |

Cada combinación (tipo × nivel) tiene un plan de intervención con acciones, plazos,
responsable, frecuencia e indicadores de seguimiento. `NORMAL` no genera intervención.

---

## Estructura del proyecto

```
api/
├── app/
│   ├── main.py                  # FastAPI: lifespan, routers, CORS, Swagger
│   ├── config.py                # Settings con pydantic-settings (.env)
│   ├── database.py              # Engine SQLAlchemy + SessionLocal
│   ├── ml/
│   │   ├── pipeline.py          # compute_features() + normalize_features()
│   │   ├── ensemble.py          # run_ensemble(): 4 modelos + score ponderado
│   │   └── classifier.py        # classify_risk / classify_type / get_intervention
│   ├── models/
│   │   └── prediction.py        # ORM: tabla predictions (PostgreSQL)
│   ├── schemas/
│   │   └── prediction.py        # Pydantic: StudentInput + PredictionResponse
│   ├── routers/
│   │   └── predict.py           # POST /predict + GET /health
│   └── services/
│       └── prediction_service.py  # Orquesta pipeline → ensemble → DB → respuesta
├── alembic/                     # Migraciones de base de datos
│   └── versions/
│       └── 0001_initial.py      # Crea tabla predictions
├── data/
│   └── RespuestasSemillero_completo.json  # Dataset crudo (81 registros)
├── models_store/                # Artefactos ML generados por train_models.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh            # Entrena si faltan artefactos → migra → uvicorn
├── train_models.py              # Entrena y serializa los 8 artefactos
├── requirements.txt
├── .env.example
└── alembic.ini
```

---

## Instalación y ejecución

### Opción A — Docker Compose (recomendada)

Requiere Docker Desktop instalado.

```bash
# 1. Clonar el repositorio y entrar al directorio
git clone <repo-url>
cd api

# 2. Crear el archivo .env a partir del ejemplo
cp .env.example .env
# Editar .env con las credenciales deseadas

# 3. Levantar los servicios (db + api)
docker compose --env-file .env -f docker/docker-compose.yml up --build
```

El `entrypoint.sh` detecta automáticamente si los artefactos ML no existen, ejecuta
`train_models.py`, aplica las migraciones de Alembic y arranca Uvicorn.

La API queda disponible en `http://localhost:8000`.

---

### Opción B — Entorno local (requiere PostgreSQL corriendo)

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar DATABASE_URL con la conexión a tu PostgreSQL local

# 4. Entrenar modelos y generar artefactos
python train_models.py

# 5. Crear la tabla en la base de datos
alembic upgrade head

# 6. Arrancar la API
uvicorn app.main:app --reload
```

---

### Documentación interactiva

Una vez arrancada la API:

| Interfaz | URL |
|---|---|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Health check | http://localhost:8000/health |

---


## Variables de entorno

Copia `.env.example` a `.env` y completa los valores:

```env
# Cadena de conexión completa (usada por la API y Alembic)
DATABASE_URL=postgresql://user:password@localhost:5432/postgres_db

# Variables usadas por el servicio PostgreSQL en Docker Compose
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=

# Rutas internas (no es necesario cambiarlas en Docker)
MODEL_STORE_PATH=./models_store
DATA_PATH=./data
```

---

## Casos de prueba

Los siguientes ejemplos pueden usarse directamente en Swagger UI (`POST /predict`):

### Caso NORMAL — estudiante sin anomalía
```json
{"Q1":1,"Q2":20,"Q3":5,"Q4":4,"Q5":4,"Q6":5,"Q7":3,"Q8":2,"Q9":8,
 "Q12":5,"Q13":5,"Q15":3,"Q16":3,"Q17":2,"Q18":1.5,"Q19":4,"Q20":6,
 "Q21":0,"Q23":4,"Q24":4,"Q28":4,"Q29":18,"Q32":3,"Q33":4,"Q34":4,"Q35":5}
```
Esperado: `risk_level: "NORMAL"`, `anomaly_score < 0.25`, `intervention: null`

### Caso MODERATE / HIGH — carga alta y bajo rendimiento
```json
{"Q1":0,"Q2":23,"Q3":3,"Q4":3,"Q5":3,"Q6":2,"Q7":7,"Q8":6,"Q9":2,
 "Q12":2,"Q13":1,"Q15":2,"Q16":2,"Q17":4,"Q18":3.5,"Q19":2,"Q20":1,
 "Q21":1,"Q23":3,"Q24":2,"Q28":6,"Q29":19,"Q32":2,"Q33":2,"Q34":3,"Q35":2}
```
Esperado: `risk_level: "HIGH"` o `"MODERATE"`, `anomaly_type: "Balance"` o `"Academic"`

### Caso CRITICAL — perfil extremo
```json
{"Q1":1,"Q2":28,"Q3":1,"Q4":1,"Q5":1,"Q6":1,"Q7":8,"Q8":8,"Q9":0,
 "Q12":0,"Q13":0,"Q15":1,"Q16":1,"Q17":5,"Q18":4.0,"Q19":1,"Q20":0,
 "Q21":1,"Q23":1,"Q24":1,"Q28":9,"Q29":22,"Q32":1,"Q33":1,"Q34":1,"Q35":1}
```
Esperado: `risk_level: "CRITICAL"`, `anomaly_score: 1.0`, `consensus_count: 4`

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Framework API | FastAPI 0.111 + Uvicorn |
| ML | scikit-learn 1.4, TensorFlow 2.16, NumPy 1.26 |
| Base de datos | PostgreSQL 16 |
| ORM / Migraciones | SQLAlchemy 2.0 + Alembic |
| Validación | Pydantic v2 |
| Contenedores | Docker + Docker Compose |
| Estilo de código | PEP 8 — verificado con flake8 |

---

## Créditos

Desarrollado por:
- Est. Elkin Stiven Contreras Rojas
- Mba. Ing. Julián Andrés Quimbayo Castro
- PhD. Mg. Ing. José Miguel Llanos Mosquera

Corporación Universitaria del Huila (CORHUILA).  
Proyecto de grado — Sistema de Detección de Anomalías Académicas.
