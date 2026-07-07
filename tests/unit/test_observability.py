"""Structured logging emits parseable single-line JSON with extras preserved."""
import io
import json
import logging

from quant_platform.observability import JsonFormatter, configure_json_logging


def make_record(**extra):
    record = logging.LogRecord("quant.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_basic_event_shape():
    line = JsonFormatter().format(make_record())
    event = json.loads(line)
    assert event["level"] == "INFO" and event["logger"] == "quant.test"
    assert event["message"] == "hello world"
    assert event["ts"].endswith("+00:00")
    assert "\n" not in line


def test_extras_preserved_and_unserializable_repr():
    event = json.loads(JsonFormatter().format(make_record(symbol="BTC-USD", conn=object())))
    assert event["symbol"] == "BTC-USD"
    assert event["conn"].startswith("<object object")


def test_exception_captured():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord("q", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    event = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in event["exception"]


def test_configure_idempotent():
    stream = io.StringIO()
    root = configure_json_logging(stream=stream)
    count = len(root.handlers)
    configure_json_logging(stream=stream)
    assert len(root.handlers) == count  # no duplicate handlers
    logging.getLogger("quant.test2").info("ping", extra={"run_id": "r1"})
    event = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert event["message"] == "ping" and event["run_id"] == "r1"
