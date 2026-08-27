import os
import sys

import pytest

_SERVICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "hermes-agent",
)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from proposal_generator import HermesProposal, ProposalGenerator, ProposedChange


class TestHermesAgent:
    def test_proposal_validation_success(self):
        change = ProposedChange(
            change_class="safe_experiment",
            parameter="adx_min",
            old_value=15,
            new_value=20,
            rationale="Test rationale"
        )
        proposal = HermesProposal(
            strategy_version="AITradingStrategy",
            problem_type="regime_mismatch",
            evidence={"sample_size": 20, "net_pnl": -0.05},
            proposed_change=change,
        )
        assert proposal.status == "pending"
        assert proposal.problem_type == "regime_mismatch"

    def test_proposal_validation_unsafe_class(self):
        with pytest.raises(ValueError) as exc:
            ProposedChange(
                change_class="leverage_change",
                parameter="leverage",
                old_value=3,
                new_value=5,
                rationale="Unsafe leverage change request"
            )
        assert "tidak diizinkan" in str(exc.value)

    def test_generator_all_rules_empty(self):
        generator = ProposalGenerator(strategy_version="AITradingStrategy")
        proposals = generator.generate_all(
            loss_summary={"sample_size": 5, "net_pnl": 0},
            regime_performance={},
            incidents=[],
            calibration={},
            previous_proposals=[]
        )
        assert len(proposals) == 0
