import os
from dotenv import load_dotenv
from typing import List, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from pydantic import BaseModel, Field
from query_contextualizer import QueryContextualizer
from prompts import route_system_prompt, rag_prompt
from sql_db_agent import SQLDBAgent
from vector_db_agent import VectorDBAgent
from web_search_agent import WebSearchAgent


class RouteDecision(BaseModel):
    destination: Literal["sql_db_agent", "vector_db_agent", "web_search_agent", "generate"] = Field(
        description="The next node to route to."
    )

class QueryProcessor:
    def __init__(self, llm : any, verbose : bool = False, maxRetry : int = 2, use_few_shot : bool = True, use_semantic_filtering : bool = True, use_metadata_filtering = True) -> None:
        load_dotenv()
        self.verbose = verbose
        self.maxRetry = maxRetry
        self.tries = 0
        self.queryContextualizer = QueryContextualizer(llm, verbose = verbose)
        self.llm = llm
        sqlDBAgent = SQLDBAgent(llm, verbose = verbose, use_few_shot = use_few_shot)
        vectorDBAgent = VectorDBAgent(llm, verbose = verbose, use_semantic_filtering = use_semantic_filtering, use_metadata_filtering = use_metadata_filtering)
        webSearchAgent = WebSearchAgent(llm, verbose=verbose)

        route_prompt = ChatPromptTemplate.from_messages(
            [("system", route_system_prompt), ("placeholder", "{messages}")]
        )

        self.route_chain = route_prompt | llm.with_structured_output(RouteDecision)

        workflow = StateGraph(MessagesState)

        workflow.add_node("router", self.router)
        workflow.add_node("sql_db_agent", sqlDBAgent.app)
        workflow.add_node("vector_db_agent", vectorDBAgent.app)
        workflow.add_node("web_search_agent", webSearchAgent.app)
        workflow.add_node("generate", self.generate)


        workflow.add_edge(START, "router")
        workflow.add_conditional_edges("router", self.route)
        workflow.add_conditional_edges("sql_db_agent", self.after_sql_db_agent)
        workflow.add_edge("vector_db_agent", "router")
        workflow.add_edge("web_search_agent", "generate")
        workflow.add_edge("generate", END)

        self.app = workflow.compile()

    def route(self, state: MessagesState) -> Literal["sql_db_agent", "vector_db_agent", "web_search_agent", "generate"]:
        if self.verbose:
            print("---ROUTING QUERY---")
        self.tries = self.tries + 1
        if self.tries > self.maxRetry:
            if self.verbose:
                print("---MAX RETRIES REACHED---")
                print("Routed to web_search_agent")
            return "web_search_agent"
        destination = self.route_chain.invoke({"messages": [state["messages"][-1]]}).destination
        if self.verbose:
            print("Routed to", destination)
        return destination
    
    def router(self, state: MessagesState) -> MessagesState:
        if isinstance(state["messages"][-1],HumanMessage):
            return state
        return {"messages" : [HumanMessage(state["messages"][-1].content)]}

    def after_sql_db_agent(self, state: MessagesState) -> Literal["router", END]:
        last_message = state["messages"][-1]

        if last_message.content == "Unable to find the required context":
            return "router"

        self.tries = 0
        return END
    
    def get_context(self, state : MessagesState) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        return last_message.content

    def generate(self, state : MessagesState) -> MessagesState:
        if self.verbose:
            print("---GENERATING FINAL RESPONSE---")
        self.tries = 0
        messages = state["messages"]
        question = messages[0].content

        docs = self.get_context(state)

        # Prompt
        prompt = PromptTemplate(
            template=rag_prompt,
            input_variables=["context", "question"],
        )

        # Chain
        rag_chain = prompt | self.llm

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    def should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def processQuery(self, question : str, chatHistory : List[Dict]) -> str:
        if self.verbose:
            print("---PROCESSING QUESTION")
            print("Question :", question)
            print("---CONTEXTUALIZING QUESTION---")
        question = self.queryContextualizer.contextualize(question, chatHistory)
        response =  self.app.invoke({"messages": [HumanMessage(content = question)]})
        return response["messages"][-1].content

if __name__ == '__main__':
    from llm_factory import create_llm
    load_dotenv()

    queryProcessor = QueryProcessor(
        create_llm(),
        verbose = True
        )
    chatHistory =[
        {
            "role" : "user",
            "content" : "How many gold medals has Michael Phelps won ?"
        },
        {
            "role" : "assistant",
            "content" : "Michael Phelps has won 23 gold medals."
        }
    ]
    question = "How many medals has Neeraj Chopra won"
    print(queryProcessor.processQuery(question, chatHistory))
