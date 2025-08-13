from fastapi import FastAPI, Request, Security
from fastapi.security.api_key import APIKeyHeader
from api.schedule import router as schedule_router
from api.swap import router as swap_router
from api.train import router as train_router
from api.healthcheck import router as healthcheck_router
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv
import os
import logging
import secrets

load_dotenv()
# env
API_KEY = os.getenv("API_KEY")
CORS_ALLOW_ORIGINS = [o for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o]
ALLOWED_HOSTS = [h for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h] or ["*"]
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "0"))  # 0 = no limit

# app
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
app = FastAPI()

# Public paths that should NOT require the API key
PUBLIC_EXACT = {
    "/openapi.json",
    "/redoc",
    "/docs",
    "/api/health/check",
}

PUBLIC_PREFIXES = (
    "/docs/",
    "/api/health/check",
)

# middlewares
if os.getenv("ENABLE_CORS") == "true":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# allowed hosts
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# basic request size guard (blocks large JSON/form bodies early)
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if MAX_BODY_BYTES > 0:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413, content={"detail": "Payload too large"}
            )
    return await call_next(request)


# API key middleware
@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    path = request.url.path

    if request.method == "OPTIONS":
        return await call_next(request)

    if path in PUBLIC_EXACT or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return await call_next(request)

    if not API_KEY:
        logging.warning("API_KEY not set; API key auth is DISABLED (dev mode).")
        return await call_next(request)

    client_key = request.headers.get("x-api-key")
    if not client_key or not secrets.compare_digest(str(client_key), str(API_KEY)):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


# get api key
def get_api_key(api_key_header: str = Security(api_key_header)):
    return api_key_header


# enable header api key
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description="Roster API",
        routes=app.routes,
    )

    # security scheme definition
    schema.setdefault("components", {}).setdefault("securitySchemes", {})[
        "ApiKeyAuth"
    ] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": "Enter your API key",
    }

    # apply security to all paths by default
    for path, methods in schema.get("paths", {}).items():
        for op in methods.values():
            # if not explicitly set, require ApiKeyAuth
            op.setdefault("security", [{"ApiKeyAuth": []}])

    # but mark health as public so Swagger doesn't show a lock on it
    if "/api/health/check" in schema.get("paths", {}):
        for op in schema["paths"]["/api/health/check"].values():
            op["security"] = []  # no API key required in docs

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

# Register routers
app.include_router(schedule_router, prefix="/api")
app.include_router(swap_router, prefix="/api")
app.include_router(train_router, prefix="/api")
app.include_router(healthcheck_router, prefix="/api")
