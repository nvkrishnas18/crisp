import json
import operator
import os
import pprint
import random
# import nest_asyncio
# nest_asyncio.apply() 

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
#  Langgraph
from langgraph.graph import END, StateGraph,START
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
# from IPython.display import Image, display
# relative imports  
from .planning_prompts import *



# groq_api="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"
groq_api="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"
llm=ChatGroq(groq_api_key=groq_api,model_name="llama-3.3-70b-versatile")


## High level planning
class planning_agent_high(BaseModel):
    taskID: int = Field(..., description="Unique number representing the task")
    task: str = Field(..., description="High level planned task explained in a line")

## detailed planning
class planning_agent_detailed(BaseModel):
    taskID: int = Field(..., description="Unique number represented the task")
    task: str = Field(..., description="High level planned task explained in a line")
    details: str = Field(..., description="In detail, plan explaining step-by-step how to execute it")
    preprocessing: Optional[str] = Field(default=None,description="It defines whether any preprocessing is required to complete that task")

class final_task(BaseModel):
    all_planed_tasks: List[planning_agent_high]

class final_detailed_task(BaseModel):
    all_planed_tasks: List[planning_agent_detailed]

class en_obj(BaseModel):
    enhanced_objective:str=Field(description="final more enhance objective")
    imp_columns : list=Field(description="List of all important columns for the user analysis from the data schema")

class GraphState(TypedDict):    
    user_query : str
    enhanced_objective :str
    imp_columns: list
    input_data : list[dict]
    preprocessing_plan:list[dict]
    stats_analysis_plan : list[dict]
    exploratory_analysis_plan : list[dict]
    visualization_plan : list[dict]
    final_plan:dict
    gen_raw_code : str

# user_query="analyze the data and provide insights"
## Nodes
def objective_enhance(GraphState):
    print("OBJECTIBVE ENHANCER")
    print(GraphState,"\n\n")
    
    GraphState["input_data"]=json.loads(GraphState["input_data"])["columns"]
    llm_structured=llm.with_structured_output(en_obj)
    response = llm_structured.invoke([SystemMessage(content=objective_system_prmt.replace("{dataschema}",str(GraphState["input_data"]))),
            HumanMessage(content=GraphState["user_query"])
        ])
        
    # result_text = response
    
#     print(result_text)
    return {"enhanced_objective":response.enhanced_objective,"imp_columns":response.imp_columns}


def preprocess_plan_agent(GraphState):    
    print("PREPROCESSING AGENT")
    llm_structured=llm.with_structured_output(final_task)
    
    response = llm_structured.invoke([SystemMessage(content=preprocess_syst_prompt.replace("{dataschema}",str(GraphState["input_data"])).replace("{enhanced_question}",GraphState["enhanced_objective"])),
            HumanMessage(content=GraphState["user_query"])
        ])
        
    result_text = response.all_planed_tasks

    final_result=[i.dict() for i in result_text]
#     print(final_result)
    return {"preprocessing_plan":final_result}


def preprocess_plan_agent_dtld(GraphState):    
    print("PREPROCESSING DETAILED AGENT")
    llm_structured=llm.with_structured_output(final_detailed_task)
    response = llm_structured.invoke([SystemMessage(content=preprocess_dtild_syst_prompt.replace("{dataschema}"
            ,str(GraphState["input_data"])).replace("{enhanced_question}",str(GraphState["enhanced_objective"]))),
            HumanMessage(content=GraphState["user_query"]+str("high-level plan:"+str(GraphState["preprocessing_plan"])))])
        
    result_text = response.all_planed_tasks

    final_result=[i.dict() for i in result_text]
    
#     print(final_result)
    return {"preprocessing_plan":final_result}

def expl_analysis_plan_agent(GraphState):    
    print("EXPLORATORY ANALYSSIS AGENT")
    print(GraphState["user_query"])
    llm_structured=llm.with_structured_output(final_task)
    response = llm_structured.invoke([SystemMessage(content=expl_analyss_syst_prompt.replace("{dataschema}",str(GraphState["input_data"])).replace("{enhanced_question}",GraphState["enhanced_objective"])),
            HumanMessage(content=GraphState["user_query"])
        ])
       
    final_result=[i.dict() for i in response.all_planed_tasks]
    
#     print(result_text)
    return {"exploratory_analysis_plan":final_result}


def expl_analysis_detaild_plan_agent(GraphState):    
    print("EXPLORATORY DETAILED ANALYSSIS AGENT")
    llm_structured=llm.with_structured_output(final_detailed_task)
#     print(GraphState["exploratory_analysis_plan"])
    
    response = llm_structured.invoke([SystemMessage(content=expl_dtld_analyss_syst_prompt.replace("{dataschema}",str(GraphState["input_data"])).replace("{enhanced_question}",GraphState["enhanced_objective"]
)),
            HumanMessage(content=GraphState["user_query"]+str("high-level plan:"+str(GraphState["exploratory_analysis_plan"])))
        ])
        
    final_result=[i.dict() for i in response.all_planed_tasks]
    
#     print(result_text)
    return {"exploratory_analysis_plan":final_result}


def stats_analysis_plan_agent(GraphState):
    print("STATISTICAL ANALYSSIS AGENT")
    llm_structured=llm.with_structured_output(final_task)
    response = llm_structured.invoke([SystemMessage(content=stats_analyss_syst_prompt.replace("{dataschema}",str(GraphState["input_data"])).replace("{enhanced_question}",GraphState["enhanced_objective"])),
            HumanMessage(content=GraphState["user_query"])
        ])
        
    final_result=[i.dict() for i in response.all_planed_tasks]
    
    print(final_result)
    return {"stats_analysis_plan":final_result}


def stats_analysis_plan_dtld_agent(GraphState):    
    print("STATISTICAL ANALYSSIS DETAILED AGENT")
    llm_structured=llm.with_structured_output(final_detailed_task)
       
    response = llm_structured.invoke([SystemMessage(content=stats_analyss_detald_syst_prompt.replace("{dataschema}",str(GraphState["input_data"])).replace("{enhanced_question}",GraphState["enhanced_objective"])),
            HumanMessage(content=GraphState["user_query"]+str("high-level plan:"+str(GraphState["stats_analysis_plan"])))])
    
    final_result=[i.dict() for i in response.all_planed_tasks]
    
#     print(final_result)
    return {"stats_analysis_plan":final_result}

def visulisation_plan_agent(GraphState):    
    print("VISULIZATION AGENT")
    llm_structured=llm.with_structured_output(final_task)
#     print("hiiii")
    response = llm_structured.invoke([SystemMessage(content=visual_analyss_syst_prompt.replace("{dataschema}",str(GraphState["input_data"])).replace("{enhanced_question}",str(GraphState["enhanced_objective"]))),
            HumanMessage(content=GraphState["user_query"])
        ])
    
    final_result=[i.dict() for i in response.all_planed_tasks]
    
    print(final_result)
    return {"visualization_plan":final_result}

def visulisation_plan_dtld_agent(GraphState):    
    print("VISULIZATION DETAILED AGENT")
    llm_structured=llm.with_structured_output(final_detailed_task)
    response = llm_structured.invoke([SystemMessage(content=visual_analyss_dtld_syst_prompt.replace("{dataschema}",str(GraphState["input_data"])).replace("{enhanced_question}",GraphState["enhanced_objective"])),
                                   HumanMessage(content=GraphState["user_query"]+str("high-level plan:"+str(GraphState["visualization_plan"])))])
        
    final_result=[i.dict() for i in response.all_planed_tasks]
    
#     print(final_result)
    return {"visualization_plan":final_result}

def code_gen(GraphState):
    print("CODE GENERATING AGENT")
#     print(GraphState)
    # response = llm.invoke([SystemMessage(content=syst_prompt_codegen),
    #         HumanMessage(content="preprocessing_detailed_plan: "+str(GraphState["preprocessing_plan"])+"\n\nexploratory_analysis_plan: "+str(GraphState["exploratory_analysis_plan"])+
    #         "\n\nstats_analysis_detailed_plan: "+str(GraphState["stats_analysis_plan"])+"\n\visualization_detailed_plan: "+str(GraphState["visualization_plan"]))
    #     ])
        
    final_result={}
    final_result["preprocessing_plan"]=GraphState["preprocessing_plan"]
    final_result["exploratory_analysis_plan"]=GraphState["exploratory_analysis_plan"]   
    final_result["stats_analysis_plan"]=GraphState["stats_analysis_plan"]
    final_result["visualization_plan"]=GraphState["visualization_plan"]
    print("Final result: ",final_result)
    return {"final_plan":final_result}