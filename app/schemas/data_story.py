from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class DataStoryCreate(BaseModel):
    """
    Schema for creating a new data story
    """
    title: str
    description: Optional[str] = None
    dataset_uuid: UUID
    analysis_objective: str


class DataStoryUpdate(BaseModel):
    """
    Schema for updating an existing data story
    """
    title: Optional[str] = None
    description: Optional[str] = None
    dataset_uuid: Optional[UUID] = None
    analysis_objective: Optional[str] = None


class DataStoryResponse(BaseModel):
    """
    Schema for data story response
    """
    story_uuid: UUID
    title: str
    description: Optional[str]
    dataset_uuid: UUID
    analysis_objective: str
    #created_time: datetime
    #created_by: str
    #updated_time: datetime
    #updated_by: str

    class Config:
        from_attributes = True


class DataStoryListResponse(BaseModel):
    """
    Schema for list of data stories
    """
    data_stories: List[DataStoryResponse]
    total: int