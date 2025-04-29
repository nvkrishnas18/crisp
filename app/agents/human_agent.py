from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from typing import Any


class State(TypedDict):
    input: str
    bot_msg: str
    user_feedback: str
    data_frame: pd.DataFrame
    input_data: dict[str, Any]


def step_1(state):
    print("---Step 1---")
    return {"bot_msg":"Please provide feedback:"}


def human_feedback(state):
    print("---human_feedback---",state)
    
    feedback = interrupt(state["bot_msg"])
    return {"user_feedback": feedback}


def step_3(state):
    print("---Step 3---", state['user_feedback'])
    # pass
    if "user_feedback" not in state:
        feedback_prompt = "Please provide feedback:"
    else:
        feedback_prompt = f"You entered :::{state["user_feedback"]}:::. Please provide feedback again:"
    df = pd.read_csv(r"D:\suresh\work\projects\W360_MVP1\playground\notebooks\data\Electric_Vehicle_Population_Data.csv")
    return {"bot_msg": feedback_prompt,"data_frame":df}

def step_4(state):
    return {"bot_msg": "Thanks for your feedback!"}

# Define the function that determines whether to continue or not
def should_continue(state):
    if state['user_feedback']=='stop':
        return 'step_4'
    else:
        return 'human_feedback'

def get_graph():
    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("step_3", step_3)
    builder.add_node("step_4", step_4)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "human_feedback")
    builder.add_edge("human_feedback", "step_3")
    # builder.add_edge("step_3", END)
    builder.add_conditional_edges("step_3",should_continue)
    builder.add_edge("step_4", END)
    # Set up memory
    memory = MemorySaver()

    # Add
    graph = builder.compile(checkpointer=memory)
    # graph = builder.compile()
    return graph

    

def get_agent():
    graph = get_graph()
    # Input
    initial_input = {"input": "Welcome to the feedback system"}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    
    return {"agent": graph, "thread": thread}
    # return get_graph()

def send_message(ws_agent,user_text):
    print("Insde send_message::begin")
    agent, thread = ws_agent["agent"], ws_agent["thread"]
    print("Insde send_message::2")
    for event in agent.stream(Command(resume=user_text), thread, stream_mode="values"):
        print(event)
        print("\n")
    print("Insde send_message::3")
    agent_state = agent.get_state(thread).values
    print("Agent_state==",agent_state)
    bot_reply = agent_state["bot_msg"]
    print("Inside send_message::bot_reply==",bot_reply)
    return bot_reply
