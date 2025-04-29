import asyncio
import logging
from uuid import UUID
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.config.config import settings
from app.models.dataset import DataSet
import os
# import nest_asyncio
# import asyncio
import requests
import httpx
# try:
#     nest_asyncio.apply()
# except Exception:
#     pass  

logger = logging.getLogger(__name__)

# def run_async(coro):
#     """
#     Run an async function 
#     """
#     try:
#         loop = asyncio.get_event_loop()
#         if loop.is_running():
#             return asyncio.run(coro)
#         else:
#             return loop.run_until_complete(coro)
#     except RuntimeError:
#         return asyncio.run(coro)

# async def get_dataset_by_uuid_async(dataset_uuid):
#     """
#     Async function to retrieve a dataset by UUID using Beanie.
    
#     """
#     try:
#         if isinstance(dataset_uuid, str):
#             uuid_obj = UUID(dataset_uuid)
#         else:
#             uuid_obj = dataset_uuid
        
#         #if settings.MONGO_HOST.startswith('localhost'):
#          #   conn_str = f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}"
#         #else:
#          #   conn_str = f"mongodb+srv://{settings.MONGO_USER}:{settings.MONGO_PASSWORD}@{settings.MONGO_HOST}"
        
#         #client = AsyncIOMotorClient(conn_str)
#         #await init_beanie(
#             #database=client[settings.MONGO_DB],
#            # document_models=[DataSet]
#         #)
        
#         dataset = await DataSet.find_one({"Dataset_uuid": uuid_obj})
        
#         if not dataset:
#             dataset = await DataSet.find_one({"Dataset_uuid": str(uuid_obj)})
        
#         #client.close()
        
#         return dataset
        
#     except Exception as e:
#         logger.error(f"Error retrieving dataset: {str(e)}")
#         return None

def get_dataset_by_uuid(dataset_uuid):
    """
    Synchronous wrapper for get_dataset_by_uuid_async.
    
    Args:
        dataset_uuid: UUID of the dataset (string or UUID object)
        
    Returns:
        DataSet: The dataset model or None if not found
    """
    try:
        ds_req_url = f"http://localhost:8000/api/v1/dataset/{dataset_uuid}"
        print("DS Request URL:::", ds_req_url)
        # with httpx.Client() as client:
        # async with httpx.AsyncClient() as client:
        #     response = client.get(ds_req_url)
        response = requests.get(ds_req_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error making GET request: {str(e)}")
        return None
    # nest_asyncio.apply()
    # return run_async(get_dataset_by_uuid_async(dataset_uuid))
    # return asyncio.run(get_dataset_by_uuid_async(dataset_uuid))

def get_file_info(dataset):
    """
    Extract file path and type from a DataSet object.
    
    Args:
        dataset: DataSet model instance
        
    Returns:
        tuple: (file_path, file_type) or (None, None) if dataset is None
    """
    if not dataset:
        return None, None
    print(f"Dataset: {type(dataset)}")
    print(f"Dataset UUID: {dataset['Dataset_uuid']}")
    print(f"Dataset File Path: {dataset['file_path']}")
    file_path = dataset['file_path']
    file_type = dataset['file_type']
    # file_path = getattr(dataset, "file_path", None)
    # file_type = getattr(dataset, "file_type", None)
    
    if file_path:
        file_path = file_path.replace('\\', '/')
    
    if not file_type and file_path and '.' in file_path:
        file_type = os.path.splitext(file_path)[1]
    
    return file_path, file_type