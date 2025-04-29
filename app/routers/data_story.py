from datetime import datetime
import os
import shutil
from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from beanie.operators import And, Or

from app.models.data_story import DataStory
from app.schemas.data_story import (
    DataStoryCreate, 
    DataStoryUpdate, 
    DataStoryResponse,
    DataStoryListResponse
)
#from app.auth.auth import get_current_active_user  


router = APIRouter()


@router.post("", response_model=DataStoryResponse)
async def create_data_story(
    data_story: DataStoryCreate,
    #current_user = Depends(get_current_active_user)
):
    """
    Create a new data story 
    
    """
    story_uuid = uuid4()
    
    new_data_story = DataStory(
        story_uuid=story_uuid,
        title=data_story.title,
        description=data_story.description,
        dataset_uuid=data_story.dataset_uuid,
        analysis_objective=data_story.analysis_objective,
        #created_by=current_user.email,
        #updated_by=current_user.email,
    )
    
    await new_data_story.create()
    return new_data_story
    


@router.get("", response_model=DataStoryListResponse)
async def list_all_data_stories(
    #current_user = Depends(get_current_active_user)
):
    """
    List ALL data stories - Shows all data stories in the database
    """
    data_stories = await DataStory.find().to_list()
    total = await DataStory.find().count()
        
    return DataStoryListResponse(data_stories=data_stories, total=total)


#@router.get("/my-stories", response_model=DataStoryListResponse)
#async def list_user_data_stories(
    #current_user = Depends(get_current_active_user)
#):
    #"""
    #List all data stories created by the authenticated user
    #"""
    #query = DataStory.created_by == current_user.email
    #data_stories = await DataStory.find(query).to_list()
    #total = await DataStory.find(query).count()
        
    #return DataStoryListResponse(data_stories=data_stories, total=total)


@router.get("/{story_uuid}", response_model=DataStoryResponse)
async def get_data_story(
    story_uuid: UUID,
    #current_user = Depends(get_current_active_user)
):
    """
    Get a specific data story by UUID - only if it belongs to the current user
    """
    data_story = await DataStory.find_one(And(
        DataStory.story_uuid == story_uuid,
        #DataStory.created_by == current_user.email
    ))
    
    if not data_story:
        raise HTTPException(status_code=404, detail="Data story not found")
    
    return data_story


@router.put("/{story_uuid}", response_model=DataStoryResponse)
async def update_data_story(
    story_uuid: UUID,
    data_story_update: DataStoryUpdate,
    #current_user = Depends(get_current_active_user)
):
    """
    Update data story metadata - only if it belongs to the current user
    """
    data_story = await DataStory.find_one(And(
        DataStory.story_uuid == story_uuid,
        #DataStory.created_by == current_user.email
    ))
    
    if not data_story:
        raise HTTPException(status_code=404, detail="Data story not found")
    
    update_data = data_story_update.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(data_story, key, value)
    
    #data_story.updated_by = current_user.email
    #data_story.updated_time = datetime.utcnow()
    
    await data_story.save()
    return data_story


@router.delete("/{story_uuid}")
async def delete_data_story(
    story_uuid: UUID,
    #current_user = Depends(get_current_active_user)
):
    """
    Delete a data story - only if it belongs to the current user
    """
    data_story = await DataStory.find_one(And(
        DataStory.story_uuid == story_uuid,
        #DataStory.created_by == current_user.email
    ))
    
    if not data_story:
        raise HTTPException(status_code=404, detail="Data story not found")
    
    await data_story.delete()
    return {"message": "Data story permanently deleted"}


@router.get("/api/data-stories/{story_uuid}/messages")
async def get_data_story_messages(
    story_uuid: UUID,
    #current_user = Depends(get_current_user)
):
    """
    Get all messages for a specific data story
    """
    data_story = await DataStory.find_one(DataStory.story_uuid == story_uuid)
    
    if not data_story:
        raise HTTPException(status_code=404, detail="Data story not found")
    
    messages = data_story.messages or []
    
    return {"messages": messages}
