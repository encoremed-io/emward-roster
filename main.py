from fastapi import FastAPI
from api.schedule import router as schedule_router
from api.swap import router as swap_router

app = FastAPI()

# Register routers
app.include_router(schedule_router, prefix="/api")
app.include_router(swap_router, prefix="/api")