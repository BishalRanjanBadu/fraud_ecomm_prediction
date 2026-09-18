"""HTTP service. One process per container: `uvicorn fraud_api.api:app --workers 1`.

/live    process is up (no I/O)                        -> liveness probe
/health  model loaded AND compatibility contract holds -> readiness / startup probe
/v1/score single payment; ?explain=true adds contributions; ?allow_fallback=true opts into the incumbent rule
No CORS middleware: this is a server-to-server API.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from . import __version__, logs, predict, registry, s3_io
from . import constants as C
from .config import Settings
from .schema import ScoreRequest, ScoreResponse
from .transform import ContractError

log = logging.getLogger("fraud_api")


def load_model(app: FastAPI) -> None:
    """Load at startup. On ANY failure the model is cleared (never keep serving a stale one)."""
    st: Settings = app.state.settings
    app.state.model = None
    try:
        s3 = app.state.s3_factory(st.region)
        m = registry.load(s3, st.bucket, st.alias_key)
    except Exception as e:  # recorded, logged with traceback, and surfaced by /health as 503
        app.state.load_error = f"{type(e).__name__}: {e}"
        log.error("model_load_failed", exc_info=True, extra={"reason": app.state.load_error,
                                                              "alias": f"s3://{st.bucket}/{st.alias_key}"})
        return
    app.state.model, app.state.load_error = m, None
    log.info("model_loaded", extra={"model_version": m.version, "family": m.family, "checks": len(m.checks),
                                    "gate_status": m.gate_status})


def create_app(s3_factory=s3_io.client) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = Settings.from_env()
        logs.setup(app.state.settings.log_level)
        load_model(app)
        yield

    app = FastAPI(title="fraud-ecomm scoring API", version=__version__, lifespan=lifespan)
    app.state.s3_factory = s3_factory
    app.state.model, app.state.load_error = None, "not loaded yet"

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info("request", extra={"request_id": rid, "method": request.method, "path": request.url.path,
                                   "status": response.status_code,
                                   "latency_ms": round(1000 * (time.perf_counter() - t0), 2),
                                   "explain": request.query_params.get("explain", "false"),
                                   "source": getattr(request.state, "source", None),
                                   "model_version": getattr(request.state, "model_version", None),
                                   "prediction_id": getattr(request.state, "prediction_id", None)})
        return response

    @app.exception_handler(ContractError)
    async def contract_error(request: Request, exc: ContractError):
        return JSONResponse(status_code=422, content={"error": "contract_violation", "field": exc.field,
                                                      "detail": exc.detail})

    @app.exception_handler(RuntimeError)
    async def scoring_error(request: Request, exc: RuntimeError):
        log.error("scoring_failed", exc_info=exc, extra={"request_id": getattr(request.state, "request_id", None)})
        return JSONResponse(status_code=500, content={"error": "scoring_failed", "detail": str(exc)})

    @app.get("/live")
    def live():
        return {"status": "alive", "service_version": __version__}

    @app.get("/health")
    def health(request: Request):
        m = request.app.state.model
        if m is None:
            return JSONResponse(status_code=503, content={"status": "not_ready",
                                                          "reason": request.app.state.load_error})
        return {"status": "ready", "model_version": m.version, "family": m.family,
                "input_family": m.input_family, "contract_version": C.CONTRACT_VERSION,
                "gate_status": m.gate_status, "checks_passed": len(m.checks), "service_version": __version__,
                "required_categoricals": predict.required_categoricals(m.p06, m.p03)}

    @app.post("/v1/score", response_model=ScoreResponse, response_model_exclude_none=False)
    def score(body: ScoreRequest, request: Request,
              explain: bool = Query(False, description="add per-feature contributions (slower)"),
              allow_fallback: bool = Query(False, description="use the incumbent rule if the model is down")):
        m = request.app.state.model
        if m is None:
            reason = request.app.state.load_error
            if allow_fallback and body.payment_risk_score is not None:
                result = predict.fallback([body], reason)[0]
            else:
                request.state.source = "unavailable"
                return JSONResponse(status_code=503, content={
                    "error": "model_unavailable", "reason": reason,
                    "fallback": "send ?allow_fallback=true with payment_risk_score to use the incumbent rule"})
        else:
            result = predict.score([body], m, explain=explain)[0]
        request.state.source = result["source"]
        request.state.model_version = result["model_version"]
        request.state.prediction_id = result["prediction_id"]
        return result

    return app


app = create_app()
