from typing import Annotated, Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from beanie import Document, Indexed
from pydantic import Field


class DataSet(Document):
    """
    Document model for storing file metadata and paths
    """
    Dataset_uuid: Annotated[UUID, Field(default_factory=uuid4), Indexed(unique=True)]
    name: str = Field(...)
    description: Optional[str] = None
    file_path: str = Field(...)
    file_type: str = Field(...)
    row_count: Optional[int] = None
    column_count: Optional[int] = None 
    #created_time: datetime = Field(default_factory=datetime.utcnow)
    #created_by: str = Field(...)
    #updated_time: datetime = Field(default_factory=datetime.utcnow)
    #updated_by: str = Field(...)

    class Settings:
        name = "DataSet"