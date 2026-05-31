from typing import Any, Literal
import re
from uuid import uuid4

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from sql_db import SQLDB
from prompts import query_check_system_prompt, query_gen_system_prompt, query_gen_few_shot_system_prompt, sql_context_prompt
from few_shots import few_shots

class SQLDBAgent:
    def __init__(self, llm: any, verbose: bool = False, maxRetry : int = 3, use_few_shot : bool = True):
        self.llm = llm
        self.verbose = verbose
        self.tries = 0
        self.maxRetry = maxRetry
        self.sqlDB = SQLDB(llm, verbose = verbose)
        tools = self.sqlDB.get_tools()
        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

        @tool("sql_db_query", description="Execute a SQL query against the database. Only the query should be given as input, without any additional text.", return_direct=True)
        def db_query_tool(query: str) -> str:
            """
            Execute a SQL query against the database and get back the result.
            If the query is not correct, an error message will be returned.
            If an error is returned, rewrite the query, check the query, and try again.
            Only the query should be given as input to this tool, without any additional text. For example, if the query is "SELECT name FROM athletes", then only "SELECT name FROM athletes" should be given as input, without any additional text like "The query is: SELECT name FROM athletes".
            """
            result = self.sqlDB.db.run_no_throw(query)
            if self.verbose:
                print(result)
            if not result:
                return "Error: Query failed. Please rewrite your query and try again."
            return result
        
        query_check_prompt = ChatPromptTemplate.from_messages(
            [("system", query_check_system_prompt), ("human", "{query}")]
        )
        self.query_check = query_check_prompt | llm

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
        
        self.context_prompt = PromptTemplate(
            template = sql_context_prompt,
            input_variables = ["tables", "schema", "query", "result"]
        )
        
        if use_few_shot:
            example_messages = [
                SystemMessagePromptTemplate.from_template(
                    f"User input: {example['input']}\nSQL query: {example['query']}"
                )
                for example in few_shots
            ]
            query_gen_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(query_gen_few_shot_system_prompt),
                    *example_messages,
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
        else:
            query_gen_prompt = ChatPromptTemplate.from_messages(
                [("system", query_gen_system_prompt), ("placeholder", "{messages}")]
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
        message = self.query_gen.invoke(
            {"messages": self._sanitize_messages_for_query_gen(state)}
        )
        if self.verbose:
            print(message)
        self.tries = self.tries + 1

        normalized_message = self._normalize_query_gen_message(message)

        # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
        tool_messages = []
        if normalized_message.tool_calls:
            for tc in normalized_message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            if self.verbose:
                print("Generated Query: ", normalized_message.content)
            if not normalized_message.content.strip():
                normalized_message = AIMessage(
                    content="Error: Empty response generated. Output a SQL query or call SubmitFinalAnswer with the final answer."
                )
            elif self._has_successful_query_result(state):
                normalized_message = AIMessage(
                    content="Final_Answer:" + normalized_message.content.strip()
                )
            elif not self._looks_like_sql(normalized_message.content):
                normalized_message = AIMessage(
                    content=(
                        "Error: Generated text was not a SQL query. Output only the SQL query to execute, "
                        "or call SubmitFinalAnswer if you already have the final answer."
                    )
                )
        if normalized_message.tool_calls and not tool_messages:
            if self._has_successful_query_result(state):
                normalized_message = AIMessage(
                    "Final_Answer:" + normalized_message.tool_calls[0]["args"]["final_answer"]
                )
            else:
                normalized_message = AIMessage(
                    content=(
                        "Error: Do not submit a final answer before executing a SQL query. "
                        "Output only the SQL query to execute."
                    )
                )
        return {"messages": [normalized_message] + tool_messages}
    
    def _sanitize_messages_for_query_gen(self, state: MessagesState) -> list[HumanMessage | AIMessage]:
        """
        Strip prior tool-call metadata from the conversation before asking the model to
        generate SQL. Some local models will incorrectly reuse earlier tool names when
        tool-call messages are present in the state.
        """
        sanitized_messages: list[HumanMessage | AIMessage] = []

        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                sanitized_messages.append(message)
            elif isinstance(message, ToolMessage):
                sanitized_messages.append(
                    HumanMessage(content=f"Tool output:\n{message.content}")
                )
            elif isinstance(message, AIMessage) and message.content:
                sanitized_messages.append(AIMessage(content=message.content))

        return sanitized_messages

    def _normalize_query_gen_message(self, message: AIMessage) -> AIMessage:
        """
        Handle provider quirks where a model puts raw SQL into SubmitFinalAnswer instead of
        returning it as plain content.
        """
        if not message.tool_calls:
            return message

        tool_call = message.tool_calls[0]
        if tool_call["name"] != "SubmitFinalAnswer":
            return message

        final_answer = tool_call["args"].get("final_answer", "").strip()
        if self._looks_like_sql(final_answer):
            return AIMessage(content=final_answer)

        return message

    def _looks_like_sql(self, text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False

        return re.match(r"^(SELECT|WITH)\b", candidate, re.IGNORECASE) is not None

    def _has_successful_query_result(self, state: MessagesState) -> bool:
        for index in range(1, len(state["messages"])):
            if (
                isinstance(state["messages"][index], ToolMessage)
                and isinstance(state["messages"][index - 1], AIMessage)
            ):
                for tool_call in state["messages"][index - 1].tool_calls:
                    if (
                        tool_call["name"] == "sql_db_query"
                        and not state["messages"][index].content.startswith("Error:")
                    ):
                        return True

        return False

    def model_check_query(self, state: MessagesState) -> dict[str, list[AIMessage]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        if self.verbose:
            print("---CHECKING QUERY---")
        original_query = state["messages"][-1].content
        response = self.query_check.invoke({"query": original_query})
        checked_query = self._extract_sql(response.content, fallback=original_query)

        if self.verbose:
            print("Checked Query:", checked_query)

        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_query",
                            "args": {"query": checked_query},
                            "id": f"sql_query_{uuid4().hex}",
                        }
                    ],
                )
            ]
        }

    def _extract_sql(self, text: str, fallback: str) -> str:
        candidate = text.strip()
        if not candidate:
            return fallback.strip()

        fenced_match = re.search(r"```(?:sql)?\s*(.*?)```", candidate, re.IGNORECASE | re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1).strip()

        lines = candidate.splitlines()
        for index, line in enumerate(lines):
            if re.match(r"^\s*(SELECT|WITH)\b", line, re.IGNORECASE):
                return "\n".join(lines[index:]).strip()

        return fallback.strip()

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
        query = None
        result = None

        for index in range(len(state["messages"]) - 1, 0, -1):
            tool_message = state["messages"][index]
            previous_message = state["messages"][index - 1]

            if not isinstance(tool_message, ToolMessage) or not isinstance(previous_message, AIMessage):
                continue

            for tool_call in previous_message.tool_calls:
                if tool_call["name"] == "sql_db_query":
                    query = tool_call["args"].get("query")
                    result = tool_message.content
                    break

            if query is not None:
                break

        if query is None or result is None:
            return ""

        return self.context_prompt.invoke({"tables" : tables, "schema" : schema, "query" : query, "result" : result})
    
    def processQuery(self, query : str) -> str:
        state = self.app.invoke({"messages": [HumanMessage(content = query)]})
        return state["messages"][-1].content
    
    def visualize(self) -> None:
        image_data = self.app.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )

        # Save the image to a file
        with open("sql_db_agent_image.png", "wb") as f:
            f.write(image_data)

if __name__ == '__main__':
    from llm_factory import create_llm

    sqlDBAgent = SQLDBAgent(
        create_llm(),
        verbose = True
    )
    # sqlDBAgent.visualize()
    question = "Name the country who won most medals in Olympics"
    response =  sqlDBAgent.processQuery(question)
    print(response)
