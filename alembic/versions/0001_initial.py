"""Initial schema: predictions table

Revision ID: 0001
Revises:
Create Date: 2026-06-01 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        # 26 raw Q columns
        sa.Column("q1", sa.Float, nullable=False),
        sa.Column("q2", sa.Float, nullable=False),
        sa.Column("q3", sa.Float, nullable=False),
        sa.Column("q4", sa.Float, nullable=False),
        sa.Column("q5", sa.Float, nullable=False),
        sa.Column("q6", sa.Float, nullable=False),
        sa.Column("q7", sa.Float, nullable=False),
        sa.Column("q8", sa.Float, nullable=False),
        sa.Column("q9", sa.Float, nullable=False),
        sa.Column("q12", sa.Float, nullable=False),
        sa.Column("q13", sa.Float, nullable=False),
        sa.Column("q15", sa.Float, nullable=False),
        sa.Column("q16", sa.Float, nullable=False),
        sa.Column("q17", sa.Float, nullable=False),
        sa.Column("q18", sa.Float, nullable=False),
        sa.Column("q19", sa.Float, nullable=False),
        sa.Column("q20", sa.Float, nullable=False),
        sa.Column("q21", sa.Float, nullable=False),
        sa.Column("q23", sa.Float, nullable=False),
        sa.Column("q24", sa.Float, nullable=False),
        sa.Column("q28", sa.Float, nullable=False),
        sa.Column("q29", sa.Float, nullable=False),
        sa.Column("q32", sa.Float, nullable=False),
        sa.Column("q33", sa.Float, nullable=False),
        sa.Column("q34", sa.Float, nullable=False),
        sa.Column("q35", sa.Float, nullable=False),
        # 10 derived features
        sa.Column("f_average_performance", sa.Float, nullable=False),
        sa.Column("f_academic_load", sa.Float, nullable=False),
        sa.Column("f_life_balance", sa.Float, nullable=False),
        sa.Column("f_psychological_stress", sa.Float, nullable=False),
        sa.Column("f_family_support", sa.Float, nullable=False),
        sa.Column("f_grade_consistency", sa.Float, nullable=False),
        sa.Column("f_responsibility_idx", sa.Float, nullable=False),
        sa.Column("f_parental_education", sa.Float, nullable=False),
        sa.Column("f_socioeconomic_risk", sa.Float, nullable=False),
        sa.Column("f_interest_gap", sa.Float, nullable=False),
        # Model flags
        sa.Column("ocsvm_flagged", sa.Boolean, nullable=False),
        sa.Column("lof_flagged", sa.Boolean, nullable=False),
        sa.Column("if_flagged", sa.Boolean, nullable=False),
        sa.Column("autoencoder_flagged", sa.Boolean, nullable=False),
        # Ensemble output
        sa.Column("consensus_count", sa.Integer, nullable=False),
        sa.Column("anomaly_score", sa.Float, nullable=False),
        sa.Column("risk_level", sa.String(10), nullable=False),
        sa.Column("anomaly_type", sa.String(50), nullable=True),
        sa.Column(
            "interventions",
            postgresql.ARRAY(sa.Text),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("predictions")
