# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompts import rag_prompt

class WebSearchAgent:
    def __init__(self, llm : any, verbose : bool = False) -> None:
        self.verbose = verbose
        load_dotenv()
        # tool = TavilySearchResults(max_results = 2)
        tool = TavilySearch(max_results = 2)
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
        self._last_context = self.get_context(state)

        # Prompt
        prompt = PromptTemplate(
            template=rag_prompt,
            input_variables=["context", "question"],
        )

        # Chain
        rag_chain = prompt | self.llm

        # Run
        response = rag_chain.invoke({"context": self._last_context, "question": query})

        return response
    
    def visualize(self) -> None:
        image_data = self.app.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )

        # Save the image to a file
        with open("web_search_image.png", "wb") as f:
            f.write(image_data)

if __name__ == '__main__':
    from llm_factory import create_llm
    load_dotenv()

    webSearchAgent = WebSearchAgent(
        create_llm(),
        verbose = True
    )
    # webSearchAgent.visualize()
    question = "What is Olympics ?"
    response =  webSearchAgent.processQuery(question)
    print(response)
