"""Switch off Guardrails AI's built-in usage telemetry.

The library sends anonymous usage traces to Guardrails' own collector by default
(``enable_metrics=True`` unless ~/.guardrailsrc says otherwise). Its telemetry object is a
process-wide singleton that builds an OTLP exporter on first use, so we create it first,
disabled, and shut its exporter down: nothing can be sent afterwards, wherever the app runs.
"""

from __future__ import annotations

_disabled = False


def disable_guardrails_telemetry() -> None:
    global _disabled
    if _disabled:
        return
    from guardrails.classes.rc import RC
    from guardrails.settings import settings
    from guardrails.utils.hub_telemetry_utils import HubTelemetry

    # No metrics, and validators run locally (Guardrails' hosted inference was retired).
    settings.rc = RC(enable_metrics=False, use_remote_inferencing=False)
    telemetry = HubTelemetry(enabled=False)
    telemetry._enabled = False
    telemetry._tracer_provider.shutdown()  # stops the exporter: spans are dropped, never sent
    _disabled = True
