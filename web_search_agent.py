from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain import hub
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
    
class WebSearchAgent:
    def __init__(self, llm : any, verbose : bool = False) -> None:
        self.verbose = verbose
        load_dotenv()
        tool = TavilySearchResults(max_results = 2)
        self.llm = llm
        self.llm_with_tool = llm.bind_tools([tool], tool_choice = 'required')
        tool_node = ToolNode([tool], name = "search")

        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", self.call_tool)
        workflow.add_node("search", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", "search")
        workflow.add_edge("search", END)

        self.app = workflow.compile()

    
    def call_tool(self, state : MessagesState) -> MessagesState:
        if self.verbose:
            print("---CALL WEB SEARCH AGENT---")
        response = self.llm_with_tool.invoke(state['messages'])
        return {"messages" : [response]}
    
    def get_context(self, state : MessagesState) -> str:
        return state["messages"][-1].content
    
    def processQuery(self, query : str) -> str:
        state = self.app.invoke({"messages": [HumanMessage(content = query)]})
        context = self.get_context(state)

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = prompt | self.llm

        # Run
        response = rag_chain.invoke({"context": context, "question": question})

        return response
    
    def visualize(self) -> None:
        image_data = self.app.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )

        # Save the image to a file
        with open("web_search_image.png", "wb") as f:
            f.write(image_data)

if __name__ == '__main__':
    from langchain_mistralai import ChatMistralAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_core.messages import HumanMessage
    import os
    load_dotenv()
    rate_limiter = InMemoryRateLimiter(
        requests_per_second = 0.5,
        check_every_n_seconds = 0.1
    )
    webSearchAgent = WebSearchAgent(
        ChatMistralAI(model_name = os.environ['MISTRAL_LLM_MODEL'],temperature=0.1, rate_limiter = rate_limiter), 
        verbose = True
    )
    # webSearchAgent.visualize()
    question = "What is Olympics ?"
    response =  webSearchAgent.processQuery(question)
    print(response)
