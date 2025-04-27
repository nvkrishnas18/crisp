from datetime import datetime
from typing_extensions import TypedDict
from app.config.config import settings
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.sqlite import SqliteSaver
#from IPython.display import Image, display
import pandas as pd
from typing import Any, Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from app.utils.database import get_dataset_by_uuid, get_file_info
from .code_exec import execute_generated_code

GROQ_API_KEY = settings.GROQ_API_KEY
llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """    
    error: str
    file_path: str
    dataset_uuid: str
    story_uuid: str
    question: str
    input_data: dict[str, Any]
    data_schema: str
    data_summary: str
    messages: Annotated[list, add_messages]
    user_query: str
    code: str
    final_results: dict[str, Any]
    final_answer: dict[str, Any]
    initial_summary_displayed: bool

def json_df(json_str)->pd.DataFrame:
    import pandas as pd
    from io import StringIO
    json_buffer = StringIO(json_str)
    df = pd.read_json(json_buffer, orient='split')
    return df

def load_data(state: GraphState):
    """
    Load data from the DataSet identified by UUID in the state.
    Uses Beanie model via utility functions to access the database.
    """
    print("Loading data with UUID:", state.get("dataset_uuid", "No UUID provided"))
    
    import pandas as pd
    import os
    # from app.utils.database import get_dataset_by_uuid, get_file_info
    
    try:
        dataset_uuid = state.get("dataset_uuid")
        if not dataset_uuid:
            print("No dataset_uuid provided in state")
            return {"error": "No dataset_uuid provided", "input_data": None}
        
        dataset = get_dataset_by_uuid(dataset_uuid)
        
        if not dataset:
            print("DataSet not found in database")
            return {"error": "DataSet not found", "input_data": None}
        
        file_path, file_type = get_file_info(dataset)
        
        if not file_path:
            print("File path is empty")
            return {"error": "File path not found", "input_data": None}
        
        # print(f"Loading data from path: {file_path}")
        
        # file_extension = file_type.lower() if file_type else ''

        # file_path = os.path.join("app", "data", "Electric_Vehicle_Population_Data_2.csv")


        print(f"Loading data from path: {file_path}")
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return {"error": f"Unsupported file type: {file_extension}", "input_data": None}
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return {"input_data": df.to_json(orient='split')}
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to load data: {str(e)}", "input_data": None}

def summarize_data(state: GraphState):
    """Generate comprehensive summary of the dataset"""
    print("--- summarize_data ---")
    df_json = state['input_data']
    df = None
    if df_json is not None:
        df = json_df(df_json)
        print("summarize_data df shape:", df.shape)
    else:
        return {"data_summary": None}
    
    summary_tables = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        desc_stats = df[numeric_cols].describe()
        summary_tables.append({
            "type": "dataframe",
            "value": desc_stats.to_json(orient='split'),
            "description": "Summary of the dataset"
        })
    
    unique_counts = df.nunique()
    summary_tables.append({
        "type": "series",
        "value": unique_counts.astype(str).to_json(orient='index'),
        "description": "Count of unique values in each column"
    })
    
    data_types = df.dtypes
    summary_tables.append({
        "type": "series",
        "value": data_types.astype(str).to_json(orient='index'),
        "description": "Data types of each column"
    })
    
    null_counts = df.isnull().sum()
    null_percentages = (df.isnull().sum() / len(df) * 100).round(2)
    null_analysis = pd.DataFrame({
        'null_count': null_counts,
        'null_percentage': null_percentages
    })
    summary_tables.append({
        "type": "dataframe",
        "value": null_analysis.to_json(orient='split'),
        "description": "Null analysis of each column"
    })
    
    if 'County' in df.columns:
        top_counties = df['County'].value_counts().head(10)
        summary_tables.append({
            "type": "series",
            "value": top_counties.astype(str).to_json(orient='index'),
            "description": "Top 10 counties by count"
        })
    
    data_summary = {
        "text_answer": f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. The columns are: {', '.join(df.columns)}.",
        "tables": summary_tables
    }
    
    return {"data_summary": data_summary}

def display_summary(state: GraphState):
    """Display the data summary to the user before taking input"""
    print("--- display_summary ---")
    
    if state.get("initial_summary_displayed", False):
        print("Summary already displayed, skipping")
        return {}
    
    data_summary = state.get("data_summary", {})
    if not data_summary:
        print("No data summary available")
        summary_message = SystemMessage(content="No data summary available.")
    else:
        summary_message = SystemMessage(content=json.dumps(data_summary))
        print("Data summary prepared for display")
    
    return {"initial_summary_displayed": True, "messages": [summary_message]}


def human_userinput(state):
    print("---human_input1---")    
    user_msg = interrupt("Enter query")
    # state['user_query']=user_msg
    # user_msg = "What is the data schema?"
    return {"user_query": user_msg, "messages": [HumanMessage(content=user_msg)]}


from pydantic import BaseModel
from typing import Literal

class ResponseFormat(BaseModel):
    status: Literal["continuation", "new_thread"] = "new_thread"
    message: str
    confidence: float = 0.0

def conversation_continuity(state: GraphState):
    """
    Use the LLM to determine if the current query represents a new conversation thread
    or if it should build on previous interactions.
    """
    previous_queries = []
    
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            previous_queries.append(msg.content)
    
    if len(previous_queries) <= 1:
        return {"is_new_thread": True}
    
    current_query = previous_queries[-1]
    previous_queries = previous_queries[:-1]  
    
    system_prompt = """
    You are a data analysis expert specializing in determining the relationship between queries in a data analysis conversation.
    
    Analyze the current query in relation to previous queries to determine if it:
    1. CONTINUES the previous analysis thread (modifying, refining, or building upon previous analysis)
    2. Starts a NEW_THREAD that's completely different from previous analysis
    
    Your response MUST be valid JSON in the following format only:
    {
        "status": "continuation" or "new_thread",
        "message": "Brief explanation of your reasoning",
        "confidence": a number between 0.0-1.0 indicating your confidence
    }
    
    Do not include any text before or after the JSON object.
    """
    
    human_prompt = f"""
    Previous queries:
    {previous_queries}
    
    Current query:
    "{current_query}"
    
    Is the current query a continuation of the previous analysis thread or a new thread?
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        result_text = response.content.strip()
        
        try:
            import json
            import re
            
            json_match = re.search(r'(\{.*\})', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            
            result = json.loads(result_text)
            
            format_check = ResponseFormat(**result)
            
            is_new_thread = format_check.status == "new_thread"
            confidence = format_check.confidence
            reasoning = format_check.message
            
            print(f"Thread determination: '{format_check.status}' (confidence: {confidence:.2f})")
            print(f"Reasoning: {reasoning}")
            
            return {"is_new_thread": is_new_thread}
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"JSON parsing error: {e}. Using text-based fallback.")
            result_text_lower = result_text.lower()
            
            if "continuation" in result_text_lower:
                print("Fallback detected: CONTINUATION")
                return {"is_new_thread": False}
            else:
                print("Fallback detected: NEW_THREAD")
                return {"is_new_thread": True}
                
    except Exception as e:
        print(f"Error in conversation continuity: {str(e)}")
        return {"is_new_thread": True}

def generate_code(state: GraphState):
    print("---generate_code---")
    data_summary = state["data_summary"]
    query = state["user_query"]
    
    continuity_result = conversation_continuity(state)
    is_new_thread = continuity_result.get("is_new_thread", True)
    
    previous_queries = []
    
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            previous_queries.append(msg.content)
    
    if not is_new_thread and len(previous_queries) > 1:
        relevant_queries = previous_queries[-6:-1] if len(previous_queries) > 5 else previous_queries[:-1]
        
        cumulative_context = "This query builds upon previous requests in the conversation.\n"
        for i, prev_query in enumerate(relevant_queries):  
            cumulative_context += f"- Previous request {i+1}: {prev_query}\n"
        
        cumulative_context += f"\nCurrent request: {query}\n"
        cumulative_context += "Your code should consider the context of previous requests when answering the current one.\n"
    else:
        cumulative_context = f"This is a NEW ANALYSIS REQUEST: {query}\n"
        cumulative_context += "Start with the original dataset for this analysis.\n"
        cumulative_context += "Do not apply filters or modifications from any previous queries.\n"
    
    system_message = (
        f"You are a professional data analyst proficient in Python programming.\n"
        f"Based on the following dataset summary:\n{data_summary}\n"
        f"{cumulative_context}"
        "Provide clean python code using pandas and plotly (only the code, nothing else) that when executed will:\n"
        "The code should contain a visualization that is well labelled.\n"
        "Assume the dataset has been read into a dataframe called df.\n" 
        "IMPORTANT INSTRUCTIONS FOR FILTERING DATA:\n"
        "1. When asked to remove or filter data, ensure you use the EXACT case of values as they appear in the dataset.\n"
        "2. For names in the column, they are likely in UPPERCASE.\n"
        "3. After filtering, verify that the filtered values are actually removed by checking the results.\n"
        "If the query involves generating a chart (e.g., bar plot), please ensure the chart displays no more than 10 categories (top or bottom).\n"
        "If the user specifically requests to show all categories, include that in the code. If no specific instruction is provided, limit the chart to the top or bottom 10 categories based on the count or value\n"
        "Make sure all the important results are saved as a dictionary with name 'final_results'. \n"
        "For plotly fig object, include type as 'figure', dataframe object as 'dataframe', series object as 'series'."
        "Each result item should be dictionary with keys value, type and description about that item. Example :{'type':'str','value':'There are 10 counties', 'description':'Count of counties'}"
        "Make sure the generated code is surrounded with try and catch . In exception handler print the error."
        f"Answer the user's question: '{query}'.\n"
        "IMPORTANT: When using plotly with Series data from value_counts(), use series.index and series.values separately to avoid errors."
    )
    print("before llm generate", system_message)
    response = llm.invoke(system_message)
    print("after llm generate", response)
    return {"code": response.content}

def prepare_answer(user_query: str, fin_results: dict, code: str):
    print("----prepare answer-----")
    if fin_results is None:
        return None
    plots = []
    text_outputs= []
    tables = []
    for k, v in fin_results.items():
        print("K::",v['type'])
        if "type" in v and v['type'].startswith('fig'):
            plots.append(v['value'])
        elif "type" in v and v['type'].lower() == 'series':
            
            ser = v
            # ser['value']=ser['value'].to_json(orient='index')
            tables.append(ser)
        elif "type" in v and v['type'].lower() == 'dataframe':
            tab = v
            # tab['value']=tab['value'].to_json(orient='split')
            tables.append(tab)
        else:
            text_outputs.append({k:v})
    
    final_answer = {}
    if len(text_outputs)>0:
        answer_prompt = f"""Generate the brief answer for the given question using the related JSON data
                        Query: {user_query}
                        JSON: {text_outputs}
                        Answer:
                        """
        resp = llm.invoke(answer_prompt)
        final_answer['text_answer']=resp.content
    else:
        final_answer['text_answer']= None
    final_answer['tables']=tables
    final_answer['visualization']=plots
    if code is not None:
        final_answer['code'] = code
    # print("final_answer############", final_answer)
    return final_answer

def get_ai_message(final_answer):
    if not final_answer:
        return {}
    
    message_txts = {}
    for k,v in final_answer.items():
        if k=='text_answer' and v is not None:
            message_txts[k]=v
        elif k=='tables' and isinstance(v,list):
            new_list=[]
            for table in v:
                tab = {tk:tv for tk,tv in table.items() if tk != 'value'}
                new_list.append(tab)
            message_txts[k]=new_list
        elif k=='code' and v is not None:
            message_txts[k]=v
        else:
            pass
    return message_txts

def execute_code(state: GraphState):
    print("-----Executing code----")
    gen_code = state["code"]
    data = json_df(state["input_data"])
    print("Original dataframe shape:", data.shape)
    if 'Make' in data.columns:
        print("Original Makes frequency (top 5):", data['Make'].value_counts().head(5))
    results = execute_generated_code(gen_code, data)
    if "df" in results:
        del results["df"]
    final_results = None
    if results is not None and 'final_results' in results:
        # final_results = convert_arrays_to_lists(results['final_results'])
        final_results = results['final_results']
        for k,v in final_results.items():
            print("K::",k,v["type"])
            if "type" in v:
                if str.lower(v["type"]) == "figure":
                    # print("$$$$$$$$$$$",k,"::::::",v)
                    final_results[k]['value']=v['value'].to_json()
                elif str.lower(v["type"]) == "series":
                    print("before conversion")
                    final_results[k]['value']=v['value'].astype(str).to_json(orient='index')
                    print("after conversion")
                elif str.lower(v["type"]) == "dataframe":
                    # print("$$$$$$$$$$$",k,"::::::",v)
                    final_results[k]['value']=v['value'].to_json(orient='split')
            else:
                if 'Series' in str(type(final_results[k]['value'])):
                    final_results[k]['type'] = 'series'
                elif 'DataFrame' in str(type(final_results[k]['value'])):
                    final_results[k]['type'] = 'dataframe'
                else:
                    final_results[k]['type'] = str(type(final_results[k]['value']))
        # print("final_results::",final_results)
    #     return {"final_results":final_results}
    
    # else:
    #     return {"final_results":""}
    final_answer = prepare_answer(state['user_query'],final_results,gen_code)
    ai_message = get_ai_message(final_answer)
    return {"final_answer":final_answer, "messages": [AIMessage(json.dumps(ai_message))]}
    
def build_graph():
    builder = StateGraph(GraphState)



    builder.add_node("load_data", load_data)
    builder.add_node("summarize_data", summarize_data)
    builder.add_node("display_summary", display_summary)
    builder.add_node("human_userinput", human_userinput)
    builder.add_node("generate_code", generate_code)
    builder.add_node("execute_code", execute_code)
    # builder.add_node("prep_ans", prep_ans)
    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "summarize_data")
    builder.add_edge("summarize_data", "display_summary")
    builder.add_edge("display_summary", "human_userinput")
    builder.add_edge("human_userinput", "generate_code")
    builder.add_edge("generate_code", "execute_code")
    # builder.add_edge("generate_code", "prep_ans")
    builder.add_edge("execute_code", "human_userinput")

    print("Initializing MemorySaver for agent...")
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    print("Graph compiled with MemorySaver.")
    # graph = builder.compile()
    return graph
    

def get_agent(story_uuid = None, dataset_uuid=None):
    """
    Creates and initializes the agent with a specified dataset UUID.
    
    Args:
        dataset_uuid (str): UUID of the dataset to work with
        
    Returns:
        dict: Dictionary containing the agent and thread
    """
    graph = build_graph()
    
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

def regenerate_results_from_code(stored_response, dataframe):
    """
    Regenerate visualization and results from stored code
    
    Args:
        stored_response (dict): The stored response containing the code
        dataframe (pd.DataFrame): The dataframe to use for regeneration
        
    Returns:
        dict: The regenerated full response
    """
    if stored_response.get("storage_type") == "initial_summary":
        return {"text_answer": stored_response.get("text_answer"), "tables": [], "visualization": []}
    
    code = stored_response.get("code")
    if not code:
        return {"text_answer": "No code available to regenerate results", "tables": [], "visualization": []}
    
    results = execute_generated_code(code, dataframe)
    if "df" in results:
        del results["df"]
    
    final_results = None
    if results is not None and 'final_results' in results:
        final_results = results['final_results']
        for k, v in final_results.items():
            if "type" in v:
                if str.lower(v["type"]) == "figure":
                    final_results[k]['value'] = v['value'].to_json()
                elif str.lower(v["type"]) == "series":
                    final_results[k]['value'] = v['value'].astype(str).to_json(orient='index')
                elif str.lower(v["type"]) == "dataframe":
                    final_results[k]['value'] = v['value'].to_json(orient='split')
            else:
                if 'Series' in str(type(final_results[k]['value'])):
                    final_results[k]['type'] = 'series'
                elif 'DataFrame' in str(type(final_results[k]['value'])):
                    final_results[k]['type'] = 'dataframe'
                else:
                    final_results[k]['type'] = str(type(final_results[k]['value']))
    
    final_answer = prepare_answer(stored_response.get("user_query"), final_results, code)
    
    return final_answer



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

