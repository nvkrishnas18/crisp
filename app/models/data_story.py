from typing import Annotated, Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from beanie import Document, Indexed
from pydantic import Field


class DataStory(Document):
    """
    Document model for storing data story information
    """
    story_uuid: Annotated[UUID, Field(default_factory=uuid4), Indexed(unique=True)]
    title: str = Field(...)
    description: Optional[str] = None
    dataset_uuid: UUID = Field(...)
    analysis_objective: str = Field(...)
    messages: List[dict] = Field(default_factory=list)
    query_count: int = Field(default=0)
    #created_time: datetime = Field(default_factory=datetime.utcnow)
    #created_by: str = Field(...)
    #updated_time: datetime = Field(default_factory=datetime.utcnow)
    #updated_by: str = Field(...)

    class Settings:
        name = "Data_Story"