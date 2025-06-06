from contextlib import asynccontextmanager

from beanie import init_beanie
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

from .auth.auth import get_hashed_password
from .config.config import settings
from .models.users import User
from .models.dataset import DataSet
from . models.data_story import DataStory
from .routers.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):

    if settings.MONGO_HOST.startswith('localhost'):
        conn_str = f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}"
    else:
        conn_str = f"mongodb+srv://{settings.MONGO_USER}:{settings.MONGO_PASSWORD}@{settings.MONGO_HOST}"

    # Setup mongoDB
    # app.state.client = AsyncIOMotorClient(
    #     settings.MONGO_HOST,
    #     settings.MONGO_PORT,
    #     username=settings.MONGO_USER,
    #     password=settings.MONGO_PASSWORD,
    # )
    app.state.client = AsyncIOMotorClient(conn_str)
    await init_beanie(
        database=app.state.client[settings.MONGO_DB], document_models=[User,DataSet,DataStory]
    )

    user = await User.find_one({"email": settings.FIRST_SUPERUSER})
    if not user:
        user = User(
            email=settings.FIRST_SUPERUSER,
            hashed_password=get_hashed_password(settings.FIRST_SUPERUSER_PASSWORD),
            is_superuser=True,
        )
        await user.create()

    # yield app
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            # See https://github.com/pydantic/pydantic/issues/7186
            # for reason of using rstrip
            str(origin).rstrip("/")
            for origin in settings.BACKEND_CORS_ORIGINS
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app.include_router(api_router, prefix=settings.API_V1_STR)
