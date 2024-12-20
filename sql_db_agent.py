from typing import Any, Literal

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from sql_db import SQLDB
from prompts import query_check_system_prompt, query_gen_system_prompt, sql_context_prompt
from few_shots import few_shots

class SQLDBAgent:
    def __init__(self, llm: any, verbose: bool = False, maxRetry : int = 3):
        self.llm = llm
        self.verbose = verbose
        self.tries = 0
        self.maxRetry = maxRetry
        self.sqlDB = SQLDB(llm, verbose = verbose)
        tools = self.sqlDB.get_tools()
        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

        @tool
        def db_query_tool(query: str) -> str:
            """
            Execute a SQL query against the database and get back the result.
            If the query is not correct, an error message will be returned.
            If an error is returned, rewrite the query, check the query, and try again.
            """
            result = self.sqlDB.db.run_no_throw(query)
            if self.verbose:
                print(result)
            if not result:
                return "Error: Query failed. Please rewrite your query and try again."
            return result
        
        query_check_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessage(content="{input}"),
                SystemMessagePromptTemplate.from_template(query_check_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.query_check = query_check_prompt | llm.bind_tools(
            [db_query_tool], tool_choice="required"
        )

        workflow = StateGraph(MessagesState)
        workflow.add_node("first_tool_call", self.first_tool_call)

        # Add nodes for the first two tools
        workflow.add_node("list_tables_tool", self.create_tool_node_with_fallback([list_tables_tool]))
        workflow.add_node("get_schema_tool", self.create_tool_node_with_fallback([get_schema_tool]))

        # Add a node for a model to choose the relevant tables based on the question and available tables
        model_get_schema = llm.bind_tools([get_schema_tool])
        workflow.add_node(
            "model_get_schema",
            lambda state: {
                "messages": [model_get_schema.invoke(state["messages"])],
            },
        )

        class SubmitFinalAnswer(BaseModel):
            """Submit the final answer to the user based on the query results."""

            final_answer: str = Field(..., description="The final answer to the user")

        example_messages = [
            SystemMessagePromptTemplate.from_template(
                f"User input: {example['input']}\nSQL query: {example['query']}"
            )
            for example in few_shots
        ]
        query_gen_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(query_gen_system_prompt),
                *example_messages,
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.context_prompt = PromptTemplate(
            template = sql_context_prompt,
            input_variables = ["tables", "schema", "query", "result"]
        )
        self.query_gen = query_gen_prompt | llm.bind_tools(
            [SubmitFinalAnswer]
        )

        workflow.add_node("query_gen", self.query_gen_node)

        # Add a node for the model to check the query before executing it
        workflow.add_node("correct_query", self.model_check_query)

        # Add node for executing the query
        workflow.add_node("execute_query", self.create_tool_node_with_fallback([db_query_tool]))

        # Specify the edges between the nodes
        workflow.add_edge(START, "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        workflow.add_conditional_edges(
            "query_gen",
            self.should_continue,
        )
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "query_gen")

        # Compile the workflow into a runnable
        self.app = workflow.compile()
    
    def first_tool_call(self, state: MessagesState) -> dict[str, list[AIMessage]]:
        if self.verbose:
            print("---FIRST TOOL CALL TO LIST TABLES---")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_list_tables",
                            "args": {},
                            "id": "tool_abcd123",
                        }
                    ],
                )
            ]
        }
    
    def query_gen_node(self, state: MessagesState):
        if self.verbose:
            print("---CALL SQL DB AGENT---")
        message = self.query_gen.invoke(state)
        self.tries = self.tries + 1

        # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            if self.verbose:
                print("Generated Query: ", message.content)
            tool_messages = []
        if message.tool_calls and not tool_messages:
            message = AIMessage("Final_Answer:" + message.tool_calls[0]["args"]["final_answer"])
        return {"messages": [message] + tool_messages}
    
    def model_check_query(self, state: MessagesState) -> dict[str, list[AIMessage]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        if self.verbose:
            print("---CHECKING QUERY---")
        state["messages"].append(HumanMessage(state["messages"][-1].content))
        response = self.query_check.invoke({"messages": [state["messages"][-1]]})
        return {"messages": [response]}

    def create_tool_node_with_fallback(self, tools: list) -> RunnableWithFallbacks[Any, dict]:
        """
        Create a ToolNode with a fallback to handle errors and surface them to the agent.
        """
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )


    def handle_tool_error(self, state: MessagesState) -> dict:
        error = state.get("error")
        if self.verbose:
            print("---ERROR---")
            print(error)
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }
    
    # Define a conditional edge to decide whether to continue or end the workflow
    def should_continue(self, state: MessagesState) -> Literal[END, "correct_query", "query_gen"]:
        messages = state["messages"]
        last_message = messages[-1]
        # If there is a tool call, then we finish
        if last_message.content.startswith("Final_Answer:"):
            state["messages"][-1].content = last_message.content.removeprefix("Final_Answer:")
            self.tries = 0
            return END
        elif self.tries >= self.maxRetry:
            if self.verbose:
                print("---MAX RETRIES REACHED---")
            state["messages"][-1].content = "Unable to find the required context"
            self.tries = 0
            return END
        elif last_message.content.startswith("Error:"):
            return "query_gen"
        else:
            return "correct_query"
    
    def get_context(self, state : MessagesState) -> str:
        tables = self.sqlDB.db.get_usable_table_names()
        schema = self.sqlDB.db.get_table_info()
        query = state["messages"][-3].tool_calls[0]["args"]["query"]
        result = state["messages"][-2].content

        return self.context_prompt.invoke({"tables" : tables, "schema" : schema, "query" : query, "result" : result})
    
    def processQuery(self, query : str) -> str:
        state = self.app.invoke({"messages": [HumanMessage(content = query)]})
        self.get_context(state)
        return state["messages"][-1].content
    
    def visualize(self) -> None:
        image_data = self.app.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )

        # Save the image to a file
        with open("sql_db_agent_image.png", "wb") as f:
            f.write(image_data)

if __name__ == '__main__':
    from langchain_mistralai import ChatMistralAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    import os
    from dotenv import load_dotenv
    load_dotenv()
    rate_limiter = InMemoryRateLimiter(
        requests_per_second = 0.3,
        check_every_n_seconds = 0.1
    )
    sqlDBAgent = SQLDBAgent(
        ChatMistralAI(
            model_name = os.environ['MISTRAL_LLM_MODEL'], 
            temperature = 0.1, 
            rate_limiter = rate_limiter
            ), 
        verbose = True
    )
    # sqlDBAgent.visualize()
    question = "Name the country who won most medals in Olympics"
    response =  sqlDBAgent.processQuery(question)
    print(response)