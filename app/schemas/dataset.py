from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class DataSetCreate(BaseModel):
    """
    Schema for creating a new dataset
    """
    name: str
    description: Optional[str] = None
    file_path: str  
    file_type: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None


class DataSetUpdate(BaseModel):
    """
    Schema for updating an existing dataset
    """
    name: Optional[str] = None
    description: Optional[str] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None


class DataSetResponse(BaseModel):
    """
    Schema for data source response
    """
    Dataset_uuid: UUID
    name: str
    description: Optional[str]
    file_path: str
    file_type: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    # row_count: int
    # column_count: int
    #created_time: datetime
    #created_by: str
    #updated_time: datetime
    #updated_by: str

    class Config:
        from_attributes = True


class DataSetListResponse(BaseModel):
    """
    Schema for list of dataset
    """
    datasets: List[DataSetResponse]
    total: int