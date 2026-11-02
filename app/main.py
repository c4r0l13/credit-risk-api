"""Credit-risk scoring service.

Endpoints
---------
GET  /health                 liveness/readiness
GET  /v1/model               model card summary (version, metrics, cutoff)
POST /v1/score               score one application
POST /v1/score/batch         score up to 1,000 applications
GET  /v1/monitoring/drift    PSI per feature over the last N scored applications
GET  /metrics                Prometheus metrics
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .drift import DriftMonitor
from .model import ModelService
from .schemas import Application, BatchRequest, BatchResponse, Decision, DriftReport, Health

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("credit-api")

DECISIONS = Counter("credit_decisions_total", "Decisions issued", ["decision", "model_version"])
LATENCY = Histogram(
    "credit_request_latency_seconds", "Request latency", ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1)
)
PD_HIST = Histogram("credit_pd", "Predicted probability of default", buckets=[i / 20 for i in range(1, 21)])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Read config at startup (not import) so the same image can serve any artifact set.
    service = ModelService(os.getenv("ARTIFACT_DIR", "artifacts"))
    app.state.model = service
    app.state.drift = DriftMonitor(service.reference, window=int(os.getenv("DRIFT_WINDOW", "2000")))
    log.info(json.dumps({"event": "model_loaded", "model_version": service.version}))
    yield


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    LATENCY.labels(endpoint=request.url.path).observe(elapsed)
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{elapsed * 1000:.1f}"
    return response


def _score(request: Request, apps: list[Application]) -> list[Decision]:
    service: ModelService = request.app.state.model
    raw = pd.DataFrame([a.to_raw() for a in apps])
    scored = service.score(raw)
    request.app.state.drift.observe(scored.features)
    out = []
    for a, p, d, r in zip(apps, scored.pd, scored.decisions, scored.reasons, strict=True):
        DECISIONS.labels(decision=d, model_version=service.version).inc()
        PD_HIST.observe(float(p))
        # Decision log: IDs and outputs only, never raw personal data.
        log.info(
            json.dumps(
                {
                    "event": "decision",
                    "application_id": a.application_id,
                    "pd": round(float(p), 4),
                    "decision": d,
                    "model_version": service.version,
                }
            )
        )
        out.append(
            Decision(
                application_id=a.application_id,
                probability_of_default=round(float(p), 4),
                decision=d,
                reasons=r if d == "decline" else [],
                model_version=service.version,
            )
        )
    return out


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    return Health(status="ok", model_version=request.app.state.model.version)


@app.get("/v1/model")
def model_info(request: Request) -> dict:
    meta = request.app.state.model.metadata
    return {k: meta[k] for k in ("model_version", "trained_at", "pd_cutoff", "metrics", "features", "data")}


@app.post("/v1/score", response_model=Decision)
def score(application: Application, request: Request) -> Decision:
    return _score(request, [application])[0]


@app.post("/v1/score/batch", response_model=BatchResponse)
def score_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    return BatchResponse(decisions=_score(request, batch.applications))


@app.get("/v1/monitoring/drift", response_model=DriftReport)
def drift(request: Request) -> DriftReport:
    return DriftReport(**request.app.state.drift.report())


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
