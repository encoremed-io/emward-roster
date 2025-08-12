from fastapi import FastAPI
from api.schedule import router as schedule_router
from api.swap import router as swap_router
from api.train import router as train_router
from api.healthcheck import router as healthcheck_router

app = FastAPI()

# Register routers
app.include_router(schedule_router, prefix="/api")
app.include_router(swap_router, prefix="/api")
app.include_router(train_router, prefix="/api")
app.include_router(healthcheck_router, prefix="/api")
