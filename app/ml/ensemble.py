import numpy as np

WEIGHTS = {
    "ocsvm": 0.40,
    "lof": 0.30,
    "iso_forest": 0.20,
    "autoencoder": 0.10,
}


def run_ensemble(
    vector: np.ndarray,
    models: dict,
    threshold: float,
) -> dict:
    ocsvm_flag = models["ocsvm"].predict(vector)[0] == -1
    lof_flag = models["lof"].predict(vector)[0] == -1
    if_flag = models["iso_forest"].predict(vector)[0] == -1

    reconstruction = models["autoencoder"].predict(vector, verbose=0)
    mse = float(np.mean(np.power(vector - reconstruction, 2)))
    ae_flag = mse > threshold

    flags = {
        "ocsvm": ocsvm_flag,
        "lof": lof_flag,
        "iso_forest": if_flag,
        "autoencoder": ae_flag,
    }

    score = sum(WEIGHTS[k] * int(v) for k, v in flags.items())
    models_flagged = [k for k, v in flags.items() if v]

    return {
        "ocsvm_flagged": flags["ocsvm"],
        "lof_flagged": flags["lof"],
        "if_flagged": flags["iso_forest"],
        "autoencoder_flagged": flags["autoencoder"],
        "anomaly_score": round(score, 4),
        "consensus_count": len(models_flagged),
        "models_flagged": models_flagged,
    }
