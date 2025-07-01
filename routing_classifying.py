#Importing Libraries:

from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm=init_chat_model(
    "gpt-4o")

#Structured output parser:

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional(therapist) or logical response. "
    )

class state(TypedDict):
    messages: Annotated[list, add_messages]
    message_type:str | None

def classify_message(state: state): #This function classifies whether the user message is emotional or logical.
    last_message = state["messages"][-1]
    classifier_llm=llm.with_structured_output(MessageClassifier)

    result=classifier_llm.invoke([
        {"role": "system",
         "content":""""Classify the user message as either:
         -'emotional : if it asks for emotional support, therapy, deals with feelings, or personal problems.
         -'logical' : if it asks for facts, information, logical analysis, or practical solutions.
         """
         },
        {"role": "user", "content": last_message.content}

         ])
    return {"message_type": result.message_type}


def router(state:state):#if it is emotional we route to the therapist agent, if it is logical we route to the logical agent.
    # We check the message_type in the state to determine the next step.
    message_type= state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    else:
        return {"next": "logical"}
    

def therapist_agent(state: state):# This function handles emotional messages and provides empathetic responses.
    # It uses the LLM to generate a compassionate reply based on the user's last message.
    last_message = state["messages"][-1]
    messages=[
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on emotional aspects of the user's message.
           show empathy, validate their feelings, and help them process their emotions.
           ask thoughoutful questions to help them explore their feelings more deeply.
           Avoid givinh logical solutions unless explicitly asked."""
        },
        {"role": "user", 
         "content": last_message.content} 
    ]
    reply= llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: state):# This function handles logical messages and provides factual, straightforward responses.
    # It uses the LLM to generate a direct reply based on the user's last message.
    last_message = state["messages"][-1]
    messages=[
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and informations
         Provide clear, concise answers based on logic and evidence.
         Do not address emotions or provide emotional support.
         Be direct and straightforward in your responses."""
        },
        {"role": "user", 
         "content": last_message.content} 
    ]
    reply= llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

graph_builder=StateGraph(state)# We create a state graph to manage the flow of the chatbot.
# We add nodes to the graph for each function we defined above.
# Each node represents a step in the conversation flow.

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

# We define the edges of the graph to connect the nodes and define the flow of the conversation.
# The edges represent the transitions between different states in the conversation.
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")   


# We add conditional edges based on the output of the router function.
# The router function determines the next step based on the message type.
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),

    {
        "therapist": "therapist",
        "logical": "logical"
    }
)
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

# We compile the graph to create a runnable chatbot.
# The compiled graph will manage the state transitions and invoke the appropriate functions based on user input.
graph=graph_builder.compile()


def run_chatbot():# This function runs the chatbot, allowing the user to interact with it.
    state={"messages": [], "message_type": None}#we take the initial state

    while True:
        user_input= input("Message:")# if the user types exit we exit
        if user_input== "exit":
            print("BYE!")
            break

        state["messages"]= state.get("messages", []) +[   #if there is no message we get an empty list, 
                                                           
                                                            #or we get the user message here.
            {"role": "user", "content": user_input}

        ]
        state= graph.invoke(state)  #we invoke the graph using the current state.

        if state.get("messages") and len(state["messages"]) >0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")# if there is a message we print the last message content.


if __name__ == "__main__":
    run_chatbot()# This is the main function that runs the chatbot when the script is executed.
    # It initializes the chatbot and starts the conversation loop.

