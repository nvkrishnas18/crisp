import json
import operator
import os
import pprint
import random
# import nest_asyncio
# nest_asyncio.apply() 
from datetime import datetime
#pydantic
from typing import Annotated, Sequence, TypedDict,Dict,List,Set,Optional
import operator
#groq llm
from langchain_groq import ChatGroq
from langchain.schema import Document
from langgraph.pregel import RetryPolicy
#Langchain document loaders and others
from langchain_core.messages import BaseMessage
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
#langchain chromadb and splitters ,embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# #  Langgraph
from langgraph.graph import END, StateGraph,START
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
# from IPython.display import Image, display
# relative imports
from .planning_agents import *
# import .data_schema as ds


class GraphState(TypedDict):
    user_query :str
    enhanced_objective :str
    imp_columns: list
    input_data : list[dict]
    preprocessing_plan:list[dict]
    stats_analysis_plan : list[dict]
    exploratory_analysis_plan : list[dict]
    visualization_plan : list[dict]
    final_plan:dict
    gen_raw_code : str


def get_planning_workflow():
    planning_workflow = StateGraph(GraphState)
    # NODES
    planning_workflow.add_node("objective_enhance",objective_enhance,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("preprocess_plan_agent",preprocess_plan_agent,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("expl_analysis_plan_agent",expl_analysis_plan_agent,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("stats_analysis_plan_agent",stats_analysis_plan_agent,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("visulisation_plan_agent",visulisation_plan_agent,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("preprocess_detail_plan_agent",preprocess_plan_agent_dtld,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("expl_analysis_detailed_plan_agent",expl_analysis_detaild_plan_agent,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("stats_analysis_detailed_plan_agent",stats_analysis_plan_dtld_agent,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("visulisation_detailed_plan_agent",visulisation_plan_dtld_agent,retry=RetryPolicy(max_attempts=5))
    planning_workflow.add_node("code_agent",code_gen,retry=RetryPolicy(max_attempts=5))

    ##EDGES
    planning_workflow.add_edge(START,"objective_enhance")
    planning_workflow.add_edge("objective_enhance","preprocess_plan_agent")
    planning_workflow.add_edge("objective_enhance","expl_analysis_plan_agent")
    planning_workflow.add_edge("objective_enhance","stats_analysis_plan_agent")
    planning_workflow.add_edge("objective_enhance","visulisation_plan_agent")

    planning_workflow.add_edge("preprocess_plan_agent","preprocess_detail_plan_agent")
    planning_workflow.add_edge("expl_analysis_plan_agent","expl_analysis_detailed_plan_agent")
    planning_workflow.add_edge("stats_analysis_plan_agent","stats_analysis_detailed_plan_agent")
    planning_workflow.add_edge("visulisation_plan_agent","visulisation_detailed_plan_agent")

    planning_workflow.add_edge("preprocess_detail_plan_agent","code_agent")
    planning_workflow.add_edge("expl_analysis_detailed_plan_agent","code_agent")
    planning_workflow.add_edge("stats_analysis_detailed_plan_agent","code_agent")
    planning_workflow.add_edge("visulisation_detailed_plan_agent","code_agent")
    planning_workflow.add_edge("code_agent",END)

    memory = MemorySaver()    
    planning_workflow_graph = planning_workflow.compile(checkpointer=memory)

    return planning_workflow_graph


# data_schema= ds.get_data_schema()
# # print("Data Schema: ",data_schema["data_schema"])

# plan=get_planning_workflow().invoke(
#     {
#         "user_objective":"I want to analyze the data and find out the important columns for my analysis",
#         "data_schema": data_schema["data_schema"]}
# )

def get_agent(story_uuid = None, dataset_uuid=None):
    """
    Creates and initializes the agent with a specified dataset UUID.
    
    Args:
        dataset_uuid (str): UUID of the dataset to work with
        
    Returns:
        dict: Dictionary containing the agent and thread
    """
    graph = get_planning_workflow()
    
    initial_input = {"dataset_uuid": dataset_uuid, "story_uuid": story_uuid}
    
    thread = {"configurable": {"thread_id": "1"}}
    
    for event in graph.stream(initial_input, thread, stream_mode="updates"):
        print("\n")
    
    return {"agent": graph, "thread": thread, "story_uuid": story_uuid}

def send_message(ws_agent, user_text):
    print("Inside send_message::begin")
    agent, thread = ws_agent["agent"], ws_agent["thread"]
    story_uuid = ws_agent.get("story_uuid")
    skip_initial_summary = ws_agent.get("skip_initial_summary", False)

    if 'configurable' not in thread or 'thread_id' not in thread['configurable']:
        thread = {"configurable": {"thread_id": "1"}}
    
    if not user_text:
        if skip_initial_summary:
            print("Skipping initial summary generation")
            return {"text_answer": "Summary skipped on reconnection", "storage_type": "skip", "version": "v0"}
            
        agent_state = agent.get_state(thread).values
        {k: (v if not isinstance(v, (list, dict)) else "...") 
           for k, v in agent_state.items()}
        if "data_summary" in agent_state and isinstance(agent_state["data_summary"], dict):
            print("Sending initial data summary")
            summary_for_storage = {
                **agent_state["data_summary"],  
                "storage_type": "initial_summary",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "v0"
            }
            
            if story_uuid:
                import asyncio
                asyncio.create_task(store_agent_response_in_datastory(story_uuid, {"storage_optimized": summary_for_storage}))
                
            return summary_for_storage
        else:
            print("No initial data summary found")
            return {"text_answer": "No data summary available", "storage_type": "initial_summary", "version": "v0"}
    
    print("Processing user query:", user_text)
    for event in agent.stream(Command(resume=user_text), thread, stream_mode="values"):
        pass
    
    agent_state = agent.get_state(thread).values
    
    storage_optimized_response = {
        "text_answer": agent_state["final_answer"].get("text_answer"),
        "code": agent_state["code"],
        "user_query": agent_state["user_query"],
        "storage_type": "query_response",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    print("Optimized response for storage created")
    
    response = agent_state["final_answer"]
    
    if story_uuid:
        import asyncio
        asyncio.create_task(store_agent_response_in_datastory(story_uuid, {"storage_optimized": storage_optimized_response}))
    
    return response


async def store_agent_response_in_datastory(story_uuid, response):
    """
    Store agent response in the DataStory document with version numbering
    
    Args:
        story_uuid (UUID or str): The story UUID
        response (dict): The response from send_message
    """
    from app.models.data_story import DataStory
    from uuid import UUID
    
    if "storage_optimized" not in response:
        print("No storage optimized response found")
        return
    
    storage_data = response["storage_optimized"]
    
    if isinstance(story_uuid, str):
        try:
            story_uuid = UUID(story_uuid)
        except ValueError:
            print(f"Invalid UUID format: {story_uuid}")
            return
    
    data_story = await DataStory.find_one(DataStory.story_uuid == story_uuid)
    
    if not data_story:
        print(f"Data story with UUID {story_uuid} not found")
        return
    
    if not hasattr(data_story, "messages") or data_story.messages is None:
        data_story.messages = []
    
    if storage_data.get("storage_type") == "query_response":
        data_story.query_count = data_story.query_count + 1 if hasattr(data_story, "query_count") else 1
        storage_data["version"] = f"v{data_story.query_count}"
    else:
        storage_data["version"] = "v0"
    
    data_story.messages.append(storage_data)
    
    await data_story.save()
    
    print(f"Stored response in data story {story_uuid} with version {storage_data['version']}")