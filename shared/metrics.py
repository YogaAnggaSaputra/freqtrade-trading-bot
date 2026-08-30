"""Shared Prometheus /metrics endpoint helper.

Minimal: exposes the default prometheus_client registry so Prometheus can
scrape without 404. Call add_metrics_endpoint(app) once per FastAPI service.
"""
from fastapi import Response

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def add_metrics_endpoint(app):
    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
