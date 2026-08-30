"""Initial migration with TimescaleDB hypertables

Revision ID: 001_initial
Revises: 
Create Date: 2026-07-30 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from decimal import Decimal

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enums
    order_status_enum = ENUM(
        'DRAFT', 'RISK_PENDING', 'RISK_REJECTED', 'RISK_APPROVED',
        'SUBMITTED', 'PARTIALLY_FILLED', 'FILLED', 'PROTECTION_PENDING',
        'PROTECTED', 'EXIT_SUBMITTED', 'CLOSED', 'UNKNOWN',
        'RECONCILIATION_REQUIRED', 'MANUAL_REVIEW',
        name='orderstatus', create_type=True
    )
    order_status_enum.create(op.get_bind(), checkfirst=True)

    order_side_enum = ENUM('buy', 'sell', name='orderside', create_type=True)
    order_side_enum.create(op.get_bind(), checkfirst=True)

    order_type_enum = ENUM('market', 'limit', 'stop_market', 'stop_limit',
                          'take_profit_market', 'take_profit_limit',
                          name='ordertype', create_type=True)
    order_type_enum.create(op.get_bind(), checkfirst=True)

    margin_mode_enum = ENUM('isolated', 'crossed', name='marginmode', create_type=True)
    margin_mode_enum.create(op.get_bind(), checkfirst=True)

    risk_decision_enum = ENUM('approved', 'rejected', name='riskdecision', create_type=True)
    risk_decision_enum.create(op.get_bind(), checkfirst=True)

    killswitch_level_enum = ENUM('yellow', 'orange', 'red', 'black', name='killswitchlevel', create_type=True)
    killswitch_level_enum.create(op.get_bind(), checkfirst=True)

    # Create tables
    op.create_table(
        'trade_dossiers',
        sa.Column('trade_id', sa.String(64), primary_key=True),
        sa.Column('strategy_version', sa.String(64), nullable=False),
        sa.Column('model_version', sa.String(64), nullable=True),
        sa.Column('config_version', sa.String(64), nullable=False),
        sa.Column('market_regime', sa.String(64), nullable=False),
        sa.Column('entry_signal', JSONB, nullable=False),
        sa.Column('feature_snapshot', JSONB, nullable=False),
        sa.Column('risk_decision', JSONB, nullable=False),
        sa.Column('approved_size', sa.Numeric(20, 8), nullable=False),
        sa.Column('entry', JSONB, nullable=False),
        sa.Column('exit', JSONB, nullable=True),
        sa.Column('sl_tp', JSONB, nullable=False),
        sa.Column('order_history', JSONB, nullable=False),
        sa.Column('fills', JSONB, nullable=False),
        sa.Column('fees_funding_slippage', JSONB, nullable=False),
        sa.Column('realized_pnl', sa.Numeric(20, 8), nullable=False),
        sa.Column('exit_reason', sa.String(128), nullable=True),
        sa.Column('loss_classification', sa.String(64), nullable=True),
        sa.Column('technical_incidents', JSONB, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('closed_at', sa.DateTime, nullable=True),
    )

    op.create_table(
        'orders',
        sa.Column('order_id', sa.String(64), primary_key=True),
        sa.Column('client_order_id', sa.String(64), unique=True, nullable=False),
        sa.Column('exchange_order_id', sa.String(64), nullable=True),
        sa.Column('trade_id', sa.String(64), sa.ForeignKey('trade_dossiers.trade_id'), nullable=False),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('side', order_side_enum, nullable=False),
        sa.Column('order_type', order_type_enum, nullable=False),
        sa.Column('status', order_status_enum, nullable=False, default='DRAFT'),
        sa.Column('amount', sa.Numeric(20, 8), nullable=False),
        sa.Column('filled', sa.Numeric(20, 8), nullable=False, default=0),
        sa.Column('price', sa.Numeric(20, 8), nullable=False),
        sa.Column('avg_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('stop_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('leverage', sa.Integer, nullable=False),
        sa.Column('margin_mode', margin_mode_enum, nullable=False),
        sa.Column('time_in_force', sa.String(8), nullable=False, default='GTC'),
        sa.Column('stop_loss', sa.Numeric(20, 8), nullable=False),
        sa.Column('take_profit', sa.Numeric(20, 8), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('exchange_timestamp', sa.DateTime, nullable=True),
        sa.Column('raw_response', JSONB, nullable=False),
    )

    op.create_table(
        'fills',
        sa.Column('fill_id', sa.String(64), primary_key=True),
        sa.Column('order_id', sa.String(64), sa.ForeignKey('orders.order_id'), nullable=False),
        sa.Column('trade_id', sa.String(64), sa.ForeignKey('trade_dossiers.trade_id'), nullable=False),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('side', order_side_enum, nullable=False),
        sa.Column('price', sa.Numeric(20, 8), nullable=False),
        sa.Column('amount', sa.Numeric(20, 8), nullable=False),
        sa.Column('fee', sa.Numeric(20, 8), nullable=False),
        sa.Column('fee_currency', sa.String(16), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('exchange_timestamp', sa.DateTime, nullable=True),
        sa.Column('trade_type', sa.String(32), nullable=False),
        sa.Column('liquidation', sa.Boolean, nullable=False, default=False),
    )

    op.create_table(
        'positions',
        sa.Column('position_id', sa.String(64), primary_key=True),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('side', order_side_enum, nullable=False),
        sa.Column('size', sa.Numeric(20, 8), nullable=False),
        sa.Column('entry_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('mark_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('leverage', sa.Integer, nullable=False),
        sa.Column('margin_mode', margin_mode_enum, nullable=False),
        sa.Column('unrealized_pnl', sa.Numeric(20, 8), nullable=False, default=0),
        sa.Column('realized_pnl', sa.Numeric(20, 8), nullable=False, default=0),
        sa.Column('liquidation_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('margin_ratio', sa.Numeric(10, 6), nullable=True),
        sa.Column('stop_loss', sa.Numeric(20, 8), nullable=True),
        sa.Column('take_profit', sa.Numeric(20, 8), nullable=True),
        sa.Column('opened_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('exchange_position_id', sa.String(64), nullable=True, unique=True),
    )

    op.create_table(
        'market_candles',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('timeframe', sa.String(16), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('open', sa.Numeric(20, 8), nullable=False),
        sa.Column('high', sa.Numeric(20, 8), nullable=False),
        sa.Column('low', sa.Numeric(20, 8), nullable=False),
        sa.Column('close', sa.Numeric(20, 8), nullable=False),
        sa.Column('volume', sa.Numeric(30, 8), nullable=False),
        sa.Column('source', sa.String(32), nullable=False, default='bitget'),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'market_snapshots',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('mark_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('index_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('last_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('bid_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('ask_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('bid_size', sa.Numeric(20, 8), nullable=False),
        sa.Column('ask_size', sa.Numeric(20, 8), nullable=False),
        sa.Column('spread', sa.Numeric(20, 8), nullable=False),
        sa.Column('funding_rate', sa.Numeric(20, 8), nullable=True),
        sa.Column('open_interest', sa.Numeric(30, 8), nullable=True),
        sa.Column('volume_24h', sa.Numeric(30, 8), nullable=True),
        sa.Column('source', sa.String(32), nullable=False, default='bitget'),
    )

    op.create_table(
        'features',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('timeframe', sa.String(16), nullable=False),
        sa.Column('feature_version', sa.String(64), nullable=False),
        sa.Column('features', JSONB, nullable=False),
        sa.Column('regime', sa.String(64), nullable=True),
        sa.Column('confidence', sa.Numeric(5, 4), nullable=True),
    )

    op.create_table(
        'predictions',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('probability', sa.Numeric(5, 4), nullable=False),
        sa.Column('confidence', sa.Numeric(5, 4), nullable=False),
        sa.Column('regime', sa.String(64), nullable=True),
        sa.Column('model_version', sa.String(64), nullable=False),
    )

    op.create_table(
        'signals',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('strategy_version', sa.String(64), nullable=False),
        sa.Column('signal_type', sa.String(32), nullable=False),
        sa.Column('reason', sa.Text, nullable=False),
        sa.Column('metadata', JSONB, nullable=False),
    )

    op.create_table(
        'trade_intents',
        sa.Column('trade_id', sa.String(64), primary_key=True),
        sa.Column('client_order_id', sa.String(64), nullable=False),
        sa.Column('strategy_version', sa.String(64), nullable=False),
        sa.Column('model_version', sa.String(64), nullable=True),
        sa.Column('config_version', sa.String(64), nullable=False),
        sa.Column('pair', sa.String(32), nullable=False),
        sa.Column('side', order_side_enum, nullable=False),
        sa.Column('order_type', order_type_enum, nullable=False),
        sa.Column('amount', sa.Numeric(20, 8), nullable=False),
        sa.Column('price', sa.Numeric(20, 8), nullable=True),
        sa.Column('stop_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('leverage', sa.Integer, nullable=False),
        sa.Column('margin_mode', margin_mode_enum, nullable=False),
        sa.Column('stop_loss', sa.Numeric(20, 8), nullable=False),
        sa.Column('take_profit', sa.Numeric(20, 8), nullable=True),
        sa.Column('timeframe', sa.String(16), nullable=False),
        sa.Column('regime', sa.String(64), nullable=True),
        sa.Column('signal_metadata', JSONB, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'risk_decisions',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('trade_id', sa.String(64), sa.ForeignKey('trade_intents.trade_id'), nullable=False),
        sa.Column('decision', risk_decision_enum, nullable=False),
        sa.Column('reason', sa.Text, nullable=False),
        sa.Column('checks', JSONB, nullable=False),
        sa.Column('approved_size', sa.Numeric(20, 8), nullable=True),
        sa.Column('timestamp', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'proposals',
        sa.Column('proposal_id', sa.String(64), primary_key=True),
        sa.Column('strategy_version', sa.String(64), nullable=False),
        sa.Column('problem_type', sa.String(64), nullable=False),
        sa.Column('evidence', JSONB, nullable=False),
        sa.Column('proposed_change', JSONB, nullable=False),
        sa.Column('expected_effect', sa.Text, nullable=False),
        sa.Column('validation_plan', sa.Text, nullable=False),
        sa.Column('rollback_condition', sa.Text, nullable=False),
        sa.Column('status', sa.String(32), nullable=False, default='pending'),
        sa.Column('experiment_id', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        'experiments',
        sa.Column('experiment_id', sa.String(64), primary_key=True),
        sa.Column('proposal_id', sa.String(64), sa.ForeignKey('proposals.proposal_id'), nullable=False),
        sa.Column('candidate_config', JSONB, nullable=False),
        sa.Column('baseline_config', JSONB, nullable=False),
        sa.Column('status', sa.String(32), nullable=False, default='pending'),
        sa.Column('metrics', JSONB, nullable=False),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
    )

    op.create_table(
        'deployments',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('deployment_id', sa.String(64), unique=True, nullable=False),
        sa.Column('strategy_version', sa.String(64), nullable=False),
        sa.Column('model_version', sa.String(64), nullable=True),
        sa.Column('config_version', sa.String(64), nullable=False),
        sa.Column('environment', sa.String(32), nullable=False),
        sa.Column('status', sa.String(32), nullable=False, default='active'),
        sa.Column('deployed_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('rolled_back_at', sa.DateTime, nullable=True),
    )

    op.create_table(
        'audit_events',
        sa.Column('event_id', sa.String(64), primary_key=True),
        sa.Column('event_type', sa.String(64), nullable=False),
        sa.Column('actor', sa.String(128), nullable=False),
        sa.Column('resource_type', sa.String(64), nullable=False),
        sa.Column('resource_id', sa.String(64), nullable=False),
        sa.Column('old_value', JSONB, nullable=True),
        sa.Column('new_value', JSONB, nullable=True),
        sa.Column('metadata', JSONB, nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'incidents',
        sa.Column('incident_id', sa.String(64), primary_key=True),
        sa.Column('incident_type', sa.String(64), nullable=False),
        sa.Column('severity', sa.String(16), nullable=False),
        sa.Column('title', sa.String(256), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('related_ids', JSONB, nullable=False),
        sa.Column('status', sa.String(32), nullable=False, default='open'),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime, nullable=True),
        sa.Column('resolved_by', sa.String(128), nullable=True),
    )

    op.create_table(
        'kill_switch_log',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('level', killswitch_level_enum, nullable=False),
        sa.Column('reason', sa.Text, nullable=False),
        sa.Column('activated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('activated_by', sa.String(128), nullable=False),
        sa.Column('auto_recover', sa.Boolean, nullable=False, default=False),
        sa.Column('recovered_at', sa.DateTime, nullable=True),
    )

    # Create indexes
    op.create_index('ix_trade_dossiers_strategy_created', 'trade_dossiers', ['strategy_version', 'created_at'])
    op.create_index('ix_trade_dossiers_regime_created', 'trade_dossiers', ['market_regime', 'created_at'])
    op.create_index('ix_orders_trade_status', 'orders', ['trade_id', 'status'])
    op.create_index('ix_orders_pair_created', 'orders', ['pair', 'created_at'])
    op.create_index('ix_fills_order_timestamp', 'fills', ['order_id', 'timestamp'])
    op.create_index('ix_fills_trade_timestamp', 'fills', ['trade_id', 'timestamp'])
    op.create_index('ix_positions_pair_updated', 'positions', ['pair', 'updated_at'])
    op.create_index('ix_market_candles_pair_tf_ts', 'market_candles', ['pair', 'timeframe', 'timestamp'])
    op.create_index('ix_market_snapshots_pair_ts', 'market_snapshots', ['pair', 'timestamp'])
    op.create_index('ix_features_pair_ts', 'features', ['pair', 'timestamp'])
    op.create_index('ix_predictions_pair_ts', 'predictions', ['pair', 'timestamp'])
    op.create_index('ix_signals_pair_ts', 'signals', ['pair', 'timestamp'])
    op.create_index('ix_audit_events_ts', 'audit_events', ['timestamp'])
    op.create_index('ix_audit_events_actor_ts', 'audit_events', ['actor', 'timestamp'])

    # Convert market_candles to TimescaleDB hypertable
    op.execute("SELECT create_hypertable('market_candles', 'timestamp', chunk_time_interval => interval '1 day', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('market_snapshots', 'timestamp', chunk_time_interval => interval '1 day', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('features', 'timestamp', chunk_time_interval => interval '1 day', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('predictions', 'timestamp', chunk_time_interval => interval '1 day', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('signals', 'timestamp', chunk_time_interval => interval '1 day', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('audit_events', 'timestamp', chunk_time_interval => interval '1 day', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('kill_switch_log', 'activated_at', chunk_time_interval => interval '1 day', if_not_exists => TRUE)")

    # Add compression policies
    op.execute("ALTER TABLE market_candles SET (timescaledb.compress, timescaledb.compress_segmentby = 'pair,timeframe')")
    op.execute("SELECT add_compression_policy('market_candles', compress_after => interval '7 days')")
    op.execute("ALTER TABLE market_snapshots SET (timescaledb.compress, timescaledb.compress_segmentby = 'pair')")
    op.execute("SELECT add_compression_policy('market_snapshots', compress_after => interval '7 days')")


def downgrade() -> None:
    # Drop compression policies
    op.execute("SELECT remove_compression_policy('market_candles', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('market_snapshots', if_exists => TRUE)")

    # Drop hypertables (convert back to regular tables)
    op.execute("SELECT drop_hypertable('market_candles', if_exists => TRUE)")
    op.execute("SELECT drop_hypertable('market_snapshots', if_exists => TRUE)")
    op.execute("SELECT drop_hypertable('features', if_exists => TRUE)")
    op.execute("SELECT drop_hypertable('predictions', if_exists => TRUE)")
    op.execute("SELECT drop_hypertable('signals', if_exists => TRUE)")
    op.execute("SELECT drop_hypertable('audit_events', if_exists => TRUE)")
    op.execute("SELECT drop_hypertable('kill_switch_log', if_exists => TRUE)")

    # Drop tables in reverse order
    op.drop_table('kill_switch_log')
    op.drop_table('incidents')
    op.drop_table('audit_events')
    op.drop_table('deployments')
    op.drop_table('experiments')
    op.drop_table('proposals')
    op.drop_table('risk_decisions')
    op.drop_table('trade_intents')
    op.drop_table('signals')
    op.drop_table('predictions')
    op.drop_table('features')
    op.drop_table('market_snapshots')
    op.drop_table('market_candles')
    op.drop_table('positions')
    op.drop_table('fills')
    op.drop_table('orders')
    op.drop_table('trade_dossiers')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS killswitchlevel")
    op.execute("DROP TYPE IF EXISTS riskdecision")
    op.execute("DROP TYPE IF EXISTS marginmode")
    op.execute("DROP TYPE IF EXISTS ordertype")
    op.execute("DROP TYPE IF EXISTS orderside")
    op.execute("DROP TYPE IF EXISTS orderstatus")