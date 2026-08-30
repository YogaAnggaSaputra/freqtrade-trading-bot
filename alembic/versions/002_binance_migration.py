"""Migrate market data source default from bitget to binance

Revision ID: 002_binance_migration
Revises: 001_initial
Create Date: 2026-08-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '002_binance_migration'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # Backfill existing rows (Bitget data marked as binance source going forward).
    op.execute("UPDATE market_candles SET source = 'binance' WHERE source = 'bitget'")
    op.execute("UPDATE market_snapshots SET source = 'binance' WHERE source = 'bitget'")

    # Change column defaults to 'binance'.
    op.alter_column('market_candles', 'source', server_default='binance', nullable=False)
    op.alter_column('market_snapshots', 'source', server_default='binance', nullable=False)


def downgrade() -> None:
    op.execute("UPDATE market_candles SET source = 'bitget' WHERE source = 'binance'")
    op.execute("UPDATE market_snapshots SET source = 'bitget' WHERE source = 'binance'")

    op.alter_column('market_candles', 'source', server_default='bitget', nullable=False)
    op.alter_column('market_snapshots', 'source', server_default='bitget', nullable=False)
