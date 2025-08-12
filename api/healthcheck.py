from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/health", tags=["Health Check"])


@router.get("/check", summary="Health Check")
def healthcheck():
    return {"status": "ok"}
