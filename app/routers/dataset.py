from datetime import datetime
import os
import shutil
from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from beanie.operators import And, Or
import pandas as pd

from app.models.dataset import DataSet
from app.schemas.dataset import (
    DataSetCreate, 
    DataSetUpdate, 
    DataSetResponse,
    DataSetListResponse
)
#from app.auth.auth import get_current_active_user  


router = APIRouter()


@router.post("/new/", response_model=DataSetResponse)
async def create_dataset_with_metadata(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    Create a new data source with file upload.
    Accepts only .csv, .xlx, .xlsx, .parquet file extension 
    Returns rows & columns count.
    """

    allowed_extensions = {".csv", ".xlx", ".xlsx", ".parquet"}

    dataset_uuid = uuid4()
    upload_dir = f"/app/data_files/{dataset_uuid}"
    os.makedirs(upload_dir, exist_ok=True)

    file_name = file.filename
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file extension")

    file_path = f"{upload_dir}/{file_name}"
    file_content = await file.read()

    with open(file_path, "wb") as f:
        f.write(file_content)

        try:
            if file_extension == ".csv":
                file_type = "CSV File"
                df = pd.read_csv(file_path)
            elif file_extension in [".xls", ".xlsx"]:
                file_type = "Excel File"
                df = pd.read_excel(file_path)
            elif file_extension == ".parquet":
                file_type = "Parquet File"
                df = pd.read_parquet(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
        
        row_count, column_count = df.shape

        new_dataset = DataSet(
            Dataset_uuid=dataset_uuid,
            name=name,
            description=description,
            file_path=file_path.replace('\\', '/'),
            file_type=file_type,
            row_count=row_count,
            column_count=column_count,
        # created_by=current_user.email,
        # updated_by=current_user.email,
        )
        await new_dataset.create()
        return new_dataset


@router.get("", response_model=DataSetListResponse)
async def list_all_datasets(
    #current_user = Depends(get_current_active_user)
):
    """
    List ALL data sources - Shows all data sources in the database
    """
    datasets = await DataSet.find().to_list()
    total = await DataSet.find().count()
        
    return DataSetListResponse(datasets=datasets, total=total)


#@router.get("/my-datasets", response_model=DataSetListResponse)
#async def list_user_datasets(
    #current_user = Depends(get_current_active_user)
#):
    #"""
    #List all data sources created by the authenticated user
    #"""
    #query = DataSet.created_by == current_user.email
    #datasets = await DataSet.find(query).to_list()
    #total = await DataSet.find(query).count()
        
    #return DataSetListResponse(datasets=datasets, total=total)


@router.get("/{dataset_uuid}", response_model=DataSetResponse)
async def get_dataset(
    dataset_uuid: UUID,
    #current_user = Depends(get_current_active_user)
):
    """
    Get a specific data source by UUID - only if it belongs to the current user
    """
    dataset = await DataSet.find_one(And(
        DataSet.Dataset_uuid == dataset_uuid,
        #DataSet.created_by == current_user.email
    ))
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    return dataset


@router.put("/{dataset_uuid}", response_model=DataSetResponse)
async def update_dataset(
    dataset_uuid: UUID,
    dataset_update: DataSetUpdate,
    #current_user = Depends(get_current_active_user)
):
    """
    Update data source metadata - only if it belongs to the current user
    """
    dataset = await DataSet.find_one(And(
        DataSet.Dataset_uuid == dataset_uuid,
        #DataSet.created_by == current_user.email
    ))
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    update_data = dataset_update.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(dataset, key, value)
    
    #dataset.updated_by = current_user.email
    # dataset.updated_time = datetime.utcnow()
    
    await dataset.save()
    return dataset


@router.delete("/{dataset_uuid}")
async def delete_dataset(
    dataset_uuid: UUID,
    permanent: bool = False,
    #current_user = Depends(get_current_active_user)
):
    """
    Delete a data source - only if it belongs to the current user
    """
    dataset = await DataSet.find_one(And(
        DataSet.Dataset_uuid == dataset_uuid,
        #DataSet.created_by == current_user.email
    ))
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    file_path = dataset.file_path.replace('\\', '/')
    if os.path.exists(file_path):
            os.remove(file_path)
            
            directory = os.path.dirname(file_path)
            if os.path.exists(directory) and not os.listdir(directory):
                os.rmdir(directory)
                
    await dataset.delete()
    return {"message": "Data source permanently deleted"}

@router.get("/view/data/{dataset_uuid}")
async def get_dataset_data(
    dataset_uuid: UUID,
    from_row: int = Query(..., description="Enter the start row number"),
    to_row: int = Query(..., description="Enter the end row number")
):
    """
    Get specific data based on UUID.

    Reads CSV, Excel, JSON, or Parquet file and returns the content between given row indices.

    """
    if from_row < 0 or to_row < from_row:
        raise HTTPException(status_code=400, detail="Invalid row range: 'from_row' must be <= 'to_row' and both >= 0")

    dataset = await DataSet.find_one(And(DataSet.Dataset_uuid == dataset_uuid))

    if not dataset:
        raise HTTPException(status_code=404, detail="Data source not found")

    file_path = dataset.file_path

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File doesn't exist!")

    file_ext = os.path.splitext(file_path)[-1].lower()

    try:
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
        elif file_ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")
        

        if from_row >= len(df):
            raise HTTPException(status_code=400, detail="Start row exceeds available data")

        to_row = min(to_row, len(df) - 1)

        data = df.iloc[from_row:to_row + 1].to_dict(orient="records")

        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

allowed_extensions = {".csv", ".xlx", ".xlsx", ".parquet"}