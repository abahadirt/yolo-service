from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.routes import router
from app.model_utils import load_model


# Load model with lifespan:
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup
    print("Lifespan: Loading model...")

    # Load Model:
    loaded_session, loaded_input_name = load_model()
    app.state.onnx_session = loaded_session
    app.state.onnx_input_name = loaded_input_name

    print(f"Lifespan: Model loaded. Session type: {type(app.state.onnx_session)}, Input name: {app.state.onnx_input_name}")

    yield

    print("Lifespan: Shutting down application...")


app = FastAPI(lifespan=lifespan)

app.include_router(router)


