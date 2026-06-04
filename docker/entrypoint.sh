#!/bin/sh
set -e

# Train models only if artifacts are missing
if [ ! -f "$MODEL_STORE_PATH/ocsvm.pkl" ]; then
    echo "Model artifacts not found — running train_models.py ..."
    python train_models.py
fi

# Apply database migrations
echo "Running Alembic migrations ..."
alembic upgrade head

# Start the API
echo "Starting MADS API ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
