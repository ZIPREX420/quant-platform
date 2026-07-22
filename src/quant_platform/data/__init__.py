"""quant_platform.data - the ONLY module that talks to external data systems.

OpenBB is consumed strictly over REST (ADR-0003): importing `openbb` here is
prohibited for licensing (AGPL) and architectural reasons.
"""

from quant_platform.data.schemas import OHLCVBar, PriceHistory
from quant_platform.data.openbb_client import OpenBBClient, OpenBBClientError

__all__ = ["OHLCVBar", "PriceHistory", "OpenBBClient", "OpenBBClientError"]
