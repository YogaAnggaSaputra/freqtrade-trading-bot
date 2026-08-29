"""Add persistent position health and post-exit regret analytics.

Revision ID: 003_roadmap_analytics
Revises: 002_binance_migration
"""
from alembic import op
import sqlalchemy as sa

revision = "003_roadmap_analytics"
down_revision = "002_binance_migration"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "position_health_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("trade_id", sa.String(64), nullable=False),
        sa.Column("pair", sa.String(32), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("health_score", sa.Numeric(8, 4), nullable=False),
        sa.Column("thesis_valid", sa.Boolean(), nullable=False),
        sa.Column("momentum_decay", sa.Numeric(8, 6), nullable=False),
        sa.Column("regime", sa.String(64), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_position_health_pair_ts", "position_health_snapshots", ["pair", "timestamp"])
    op.create_index("ix_position_health_trade_id", "position_health_snapshots", ["trade_id"])
    op.create_table(
        "exit_regrets",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("trade_id", sa.String(64), nullable=False),
        sa.Column("pair", sa.String(32), nullable=False),
        sa.Column("exit_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("best_future_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("regret_pct", sa.Numeric(12, 8), nullable=False),
        sa.Column("classification", sa.String(32), nullable=False),
        sa.Column("horizon_candles", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trade_id"),
    )
    op.create_index("ix_exit_regrets_trade_id", "exit_regrets", ["trade_id"])

def downgrade() -> None:
    op.drop_index("ix_exit_regrets_trade_id", table_name="exit_regrets")
    op.drop_table("exit_regrets")
    op.drop_index("ix_position_health_trade_id", table_name="position_health_snapshots")
    op.drop_index("ix_position_health_pair_ts", table_name="position_health_snapshots")
    op.drop_table("position_health_snapshots")
