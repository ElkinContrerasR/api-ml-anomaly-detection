import uuid
from sqlalchemy.orm import Session
from app.ml import pipeline, ensemble, classifier
from app.models.prediction import Prediction
from app.schemas.prediction import (
    StudentInput,
    PredictionResponse,
    InterventionDetail,
)


def create_prediction(
    form_data: StudentInput,
    db: Session,
    models: dict,
) -> PredictionResponse:
    raw_q = form_data.model_dump()

    features = pipeline.compute_features(raw_q)
    vector = pipeline.normalize_features(features, models["scaler"])

    result = ensemble.run_ensemble(
        vector, models, models["autoencoder_threshold"]
    )
    risk_level = classifier.classify_risk(result["anomaly_score"])

    anomaly_type = None
    intervention_data = None
    if risk_level != "NORMAL":
        anomaly_type = classifier.classify_type(
            raw_q, models["semillero_means"], models["semillero_stds"]
        )
        intervention_data = classifier.get_intervention(anomaly_type, risk_level)

    prediction_id = uuid.uuid4()
    record = Prediction(
        id=prediction_id,
        q1=raw_q["Q1"], q2=raw_q["Q2"], q3=raw_q["Q3"],
        q4=raw_q["Q4"], q5=raw_q["Q5"], q6=raw_q["Q6"],
        q7=raw_q["Q7"], q8=raw_q["Q8"], q9=raw_q["Q9"],
        q12=raw_q["Q12"], q13=raw_q["Q13"],
        q15=raw_q["Q15"], q16=raw_q["Q16"],
        q17=raw_q["Q17"], q18=raw_q["Q18"], q19=raw_q["Q19"],
        q20=raw_q["Q20"], q21=raw_q["Q21"],
        q23=raw_q["Q23"], q24=raw_q["Q24"],
        q28=raw_q["Q28"], q29=raw_q["Q29"],
        q32=raw_q["Q32"], q33=raw_q["Q33"],
        q34=raw_q["Q34"], q35=raw_q["Q35"],
        f_average_performance=features["F_Average_Performance"],
        f_academic_load=features["F_Academic_Load"],
        f_life_balance=features["F_Life_Balance"],
        f_psychological_stress=features["F_Psychological_Stress"],
        f_family_support=features["F_Family_Support"],
        f_grade_consistency=features["F_Grade_Consistency"],
        f_responsibility_idx=features["F_Responsibility_Result_Index"],
        f_parental_education=features["F_Parental_Education"],
        f_socioeconomic_risk=features["F_Socioeconomic_Risk"],
        f_interest_gap=features["F_Interest_Performance_Gap"],
        ocsvm_flagged=result["ocsvm_flagged"],
        lof_flagged=result["lof_flagged"],
        if_flagged=result["if_flagged"],
        autoencoder_flagged=result["autoencoder_flagged"],
        consensus_count=result["consensus_count"],
        anomaly_score=result["anomaly_score"],
        risk_level=risk_level,
        anomaly_type=anomaly_type,
        interventions=intervention_data["actions"] if intervention_data else None,
    )
    db.add(record)
    db.commit()

    intervention = (
        InterventionDetail(**intervention_data) if intervention_data else None
    )
    return PredictionResponse(
        prediction_id=str(prediction_id),
        anomaly_score=result["anomaly_score"],
        risk_level=risk_level,
        anomaly_type=anomaly_type,
        consensus_count=result["consensus_count"],
        models_flagged=result["models_flagged"],
        intervention=intervention,
        derived_features={k: round(v, 4) for k, v in features.items()},
    )
