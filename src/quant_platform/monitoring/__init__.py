"""quant_platform.monitoring - status probes and observability.

Exports are lazy so that `python -m quant_platform.monitoring.status` does not
trigger the runpy double-import warning (the module must not be pre-imported
by its own package when executed as __main__).
"""

__all__ = ["StatusCheck", "run_all"]


def __getattr__(name):
    if name in __all__:
        from quant_platform.monitoring import status

        return getattr(status, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
