import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.main import app


HTTP_REQUESTS = Counter(
    "ai_http_requests_total",
    "Total AI model HTTP requests",
    ("method", "path", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "ai_http_request_duration_seconds",
    "AI model HTTP request duration in seconds",
    ("method", "path"),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "ai_http_requests_in_progress",
    "AI model HTTP requests currently in progress",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        HTTP_REQUESTS_IN_PROGRESS.inc()
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except BaseException:
            status = "500"
            raise
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            method = request.method
            duration = time.perf_counter() - start
            HTTP_REQUESTS.labels(method=method, path=path, status=status).inc()
            HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration)
            HTTP_REQUESTS_IN_PROGRESS.dec()

        return response


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.add_middleware(PrometheusMiddleware)
