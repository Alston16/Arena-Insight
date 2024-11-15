import os
from dotenv import load_dotenv
from typing import List, Dict, Literal
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph, MessagesState
from query_contextualizer import QueryContextualizer
from prompts import route_system_prompt
from sql_db_agent import SQLDBAgent
from vector_db_agent import VectorDBAgent
from web_search_agent import WebSearchAgent

class QueryProcessor:
    def __init__(self, llm : any, verbose : bool = False) -> None:
        load_dotenv()
        self.verbose = verbose
        self.queryContextualizer = QueryContextualizer(llm, verbose = verbose)
        self.llm = llm
        sqlDBAgent = SQLDBAgent(llm, verbose = verbose)
        vectorDBAgent = VectorDBAgent(llm, verbose = verbose)
        webSearchAgent = WebSearchAgent(llm, verbose=verbose)

        route_prompt = ChatPromptTemplate.from_messages(
            [("system", route_system_prompt), ("placeholder", "{messages}")]
        )

        self.route_chain = route_prompt | llm | StrOutputParser()

        workflow = StateGraph(MessagesState)

        workflow.add_node("router", self.router)
        workflow.add_node("sql_db_agent", sqlDBAgent.app)
        workflow.add_node("vector_db_agent", vectorDBAgent.app)
        workflow.add_node("web_search_agent", webSearchAgent.app)
        workflow.add_node("generate", self.generate)


        workflow.add_edge(START, "router")
        workflow.add_conditional_edges("router", self.route)
        workflow.add_edge("sql_db_agent", "router")
        workflow.add_edge("vector_db_agent", "router")
        workflow.add_edge("web_search_agent", "generate")
        workflow.add_edge("generate", END)

        self.app = workflow.compile()

    def route(self, state: MessagesState) -> Literal["sql_db_agent", "vector_db_agent", "web_search_agent", "generate"]:
        if self.verbose:
            print("---ROUTING QUERY---")
        destination = self.route_chain.invoke({"messages": [state["messages"][-1]]})
        if self.verbose:
            print("Routed to", destination)
        return destination
    
    def router(self, state: MessagesState) -> MessagesState:
        if isinstance(state["messages"][-1],HumanMessage):
            return state
        return {"messages" : [HumanMessage(state["messages"][-1].content)]}
    
    def get_context(self, state : MessagesState) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        return last_message.content

    def generate(self, state : MessagesState) -> MessagesState:
        if self.verbose:
            print("---GENERATING FINAL RESPONSE---")
        messages = state["messages"]
        question = messages[0].content

        docs = self.get_context(state)

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = prompt | self.llm

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    def should_continue(self, state: MessagesState) -> Literal["tools", END]:
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
    from langchain_mistralai import ChatMistralAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    load_dotenv()
    rate_limiter = InMemoryRateLimiter(
        requests_per_second = 0.3,
        check_every_n_seconds = 0.1
    )
    queryProcessor = QueryProcessor(
        ChatMistralAI(model_name = os.environ['MISTRAL_LLM_MODEL'],temperature=0.1, rate_limiter = rate_limiter), 
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