"""Persist 48-hour champion/challenger shadow evaluation state.

Revision ID: 004_model_shadow_evaluations
Revises: 003_roadmap_analytics
"""
from alembic import op
import sqlalchemy as sa

revision = "004_model_shadow_evaluations"
down_revision = "003_roadmap_analytics"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "model_shadow_evaluations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("version_id", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ends_at", sa.DateTime(), nullable=False),
        sa.Column("candidate_score", sa.Numeric(12, 8), nullable=True),
        sa.Column("champion_score", sa.Numeric(12, 8), nullable=True),
        sa.Column("samples", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["version_id"], ["model_versions.version_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_id"),
    )
    op.create_index("ix_model_shadow_status_ends", "model_shadow_evaluations", ["status", "ends_at"])

def downgrade() -> None:
    op.drop_index("ix_model_shadow_status_ends", table_name="model_shadow_evaluations")
    op.drop_table("model_shadow_evaluations")
