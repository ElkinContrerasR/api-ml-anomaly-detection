import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, Boolean, DateTime, ARRAY, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # 26 raw Q variables
    q1: Mapped[float] = mapped_column(Float, nullable=False)
    q2: Mapped[float] = mapped_column(Float, nullable=False)
    q3: Mapped[float] = mapped_column(Float, nullable=False)
    q4: Mapped[float] = mapped_column(Float, nullable=False)
    q5: Mapped[float] = mapped_column(Float, nullable=False)
    q6: Mapped[float] = mapped_column(Float, nullable=False)
    q7: Mapped[float] = mapped_column(Float, nullable=False)
    q8: Mapped[float] = mapped_column(Float, nullable=False)
    q9: Mapped[float] = mapped_column(Float, nullable=False)
    q12: Mapped[float] = mapped_column(Float, nullable=False)
    q13: Mapped[float] = mapped_column(Float, nullable=False)
    q15: Mapped[float] = mapped_column(Float, nullable=False)
    q16: Mapped[float] = mapped_column(Float, nullable=False)
    q17: Mapped[float] = mapped_column(Float, nullable=False)
    q18: Mapped[float] = mapped_column(Float, nullable=False)
    q19: Mapped[float] = mapped_column(Float, nullable=False)
    q20: Mapped[float] = mapped_column(Float, nullable=False)
    q21: Mapped[float] = mapped_column(Float, nullable=False)
    q23: Mapped[float] = mapped_column(Float, nullable=False)
    q24: Mapped[float] = mapped_column(Float, nullable=False)
    q28: Mapped[float] = mapped_column(Float, nullable=False)
    q29: Mapped[float] = mapped_column(Float, nullable=False)
    q32: Mapped[float] = mapped_column(Float, nullable=False)
    q33: Mapped[float] = mapped_column(Float, nullable=False)
    q34: Mapped[float] = mapped_column(Float, nullable=False)
    q35: Mapped[float] = mapped_column(Float, nullable=False)

    # 10 derived features
    f_average_performance: Mapped[float] = mapped_column(Float, nullable=False)
    f_academic_load: Mapped[float] = mapped_column(Float, nullable=False)
    f_life_balance: Mapped[float] = mapped_column(Float, nullable=False)
    f_psychological_stress: Mapped[float] = mapped_column(Float, nullable=False)
    f_family_support: Mapped[float] = mapped_column(Float, nullable=False)
    f_grade_consistency: Mapped[float] = mapped_column(Float, nullable=False)
    f_responsibility_idx: Mapped[float] = mapped_column(Float, nullable=False)
    f_parental_education: Mapped[float] = mapped_column(Float, nullable=False)
    f_socioeconomic_risk: Mapped[float] = mapped_column(Float, nullable=False)
    f_interest_gap: Mapped[float] = mapped_column(Float, nullable=False)

    # Model flags
    ocsvm_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    lof_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    if_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    autoencoder_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Ensemble output
    consensus_count: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    anomaly_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    interventions: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text), nullable=True
    )
