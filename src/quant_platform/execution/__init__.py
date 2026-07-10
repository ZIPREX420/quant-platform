"""quant_platform.execution - paper-first execution (live mode does not exist).

Live trading requires the Phase 10 gate (workspace risk R-4) and would be a
separate, reviewed addition; nothing in this package can reach a real venue.
"""

from quant_platform.execution.paper import (
    ExecutionMode,
    PaperAccount,
    PaperExchange,
    PaperFill,
)
from quant_platform.execution.session import AuditRecord, ExecutionAudit, PaperTradingSession
from quant_platform.execution.state import (
    OpenPosition,
    PaperState,
    StateError,
    StateStore,
)

__all__ = [
    "ExecutionMode",
    "PaperAccount",
    "PaperExchange",
    "PaperFill",
    "AuditRecord",
    "ExecutionAudit",
    "PaperTradingSession",
    "OpenPosition",
    "PaperState",
    "StateError",
    "StateStore",
]
