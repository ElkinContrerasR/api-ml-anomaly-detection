import numpy as np

FEATURE_ORDER = [
    "F_Average_Performance",
    "F_Academic_Load",
    "F_Life_Balance",
    "F_Psychological_Stress",
    "F_Family_Support",
    "F_Grade_Consistency",
    "F_Responsibility_Result_Index",
    "F_Parental_Education",
    "F_Socioeconomic_Risk",
    "F_Interest_Performance_Gap",
]


def compute_features(q: dict) -> dict[str, float]:
    f1 = (q["Q33"] + q["Q34"] + q["Q35"]) / 3
    f2 = q["Q7"] + q["Q8"] + q["Q9"]
    f3 = q["Q20"] / (f2 + 1)
    f4 = (q["Q17"] + q["Q18"]) / 2
    f5 = q["Q12"] + q["Q13"]
    f6 = float(np.std([q["Q33"], q["Q34"], q["Q35"]], ddof=1))
    f7 = q["Q3"] / (f1 + 1)
    f8 = q["Q15"] + q["Q16"]
    f9 = (6 - q["Q32"]) + (1 - q["Q21"])
    f10 = abs(q["Q6"] - (f1 / 3))
    return {
        "F_Average_Performance": round(f1, 6),
        "F_Academic_Load": round(f2, 6),
        "F_Life_Balance": round(f3, 6),
        "F_Psychological_Stress": round(f4, 6),
        "F_Family_Support": round(f5, 6),
        "F_Grade_Consistency": round(f6, 6),
        "F_Responsibility_Result_Index": round(f7, 6),
        "F_Parental_Education": round(f8, 6),
        "F_Socioeconomic_Risk": round(f9, 6),
        "F_Interest_Performance_Gap": round(f10, 6),
    }


def normalize_features(features: dict[str, float], scaler) -> np.ndarray:
    vector = np.array(
        [features[k] for k in FEATURE_ORDER], dtype=np.float64
    ).reshape(1, -1)
    return scaler.transform(vector)
