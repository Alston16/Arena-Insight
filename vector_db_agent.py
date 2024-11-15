from typing import Literal
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_huggingface import HuggingFaceEmbeddings
from vector_db import VectorDB
from prompts import grade_document_prompt
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

class VectorDBAgent:
    def __init__(self, llm : any, verbose : bool = False, maxRetry : int = 3) -> None:
        load_dotenv()
        self.llm = llm
        self.verbose = verbose
        self.maxRetry = maxRetry
        self.tries = 0
        vectorDB = VectorDB(
            os.environ['VECTOR_DB_DIRECTORY'], 
            HuggingFaceEmbeddings(model_name=os.environ['EMBEDDINGS_MODEL']), 
            verbose = verbose
            )
        self.tools = [vectorDB.as_tool()]

        # Define a new graph
        workflow = StateGraph(MessagesState)

        # Define the nodes we will cycle between
        # agent
        workflow.add_node("agent", self.agent)  
        retrieve = ToolNode(self.tools)
        # retrieval
        workflow.add_node("retrieve", retrieve) 
        # Re-writing the question 
        workflow.add_node("rewrite", self.rewrite)  
        # # Generating a response after we know the documents are relevant
        workflow.add_node("generate", self.generate)  
        # Call agent node to decide to retrieve or not
        workflow.add_edge(START, "agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.grade_documents,
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        # Compile
        self.app = workflow.compile()

    def grade_documents(self, state: MessagesState) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        if self.verbose:
            print("---CHECK RELEVANCE---")

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM with tool and validation
        llm_with_tool = self.llm.with_structured_output(grade)

        # Prompt
        prompt = PromptTemplate(
            template = grade_document_prompt,
            input_variables = ["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score

        if score == "yes":
            if self.verbose:
                print("---DECISION: DOCS RELEVANT---")
            return "generate"
        
        elif self.tries >= self.maxRetry:
            if self.verbose:
                print("---MAX RETRIES REACHED---")
            return "generate"

        else:
            if self.verbose:
                print("---DECISION: DOCS NOT RELEVANT---")
                print(score)
            return "rewrite"
    
    def agent(self, state : MessagesState) -> MessagesState:
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        if self.verbose:
            print("---CALL VECTOR DB AGENT---")
        messages = state["messages"]
        self.tries = self.tries + 1
        if isinstance(messages[-1], AIMessage):
            messages.append(HumanMessage(messages[-1].content))
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        return {"messages": [response]}
    
    def rewrite(self, state : MessagesState) -> MessagesState:
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        if self.verbose:
            print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: 
        Only return the new question and no additional context.""",
            )
        ]

        response = self.llm.invoke(msg)
        if self.verbose:
            print("Transformed Query :", response.content)
        return {"messages": [response]}
    
    def generate(self, state : MessagesState) -> MessagesState:
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        if self.verbose:
            print("---GENERATE---")
        messages = state["messages"]
        self.tries = 0
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = prompt | self.llm

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    def visualize(self) -> None:
        image_data = self.app.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )

        # Save the image to a file
        with open("vector_db_agent_image.png", "wb") as f:
            f.write(image_data)

if __name__ == '__main__':
    from langchain_mistralai import ChatMistralAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    load_dotenv()
    rate_limiter = InMemoryRateLimiter(
        requests_per_second = 0.5,
        check_every_n_seconds = 0.1
    )
    vectorDBAgent = VectorDBAgent(
        ChatMistralAI(
            model_name = os.environ['MISTRAL_LLM_MODEL'], 
            temperature = 0.1, 
            rate_limiter = rate_limiter
            ), 
        verbose = True
        )
    # vectorDBAgent.visualize()
    question = "How to score a three pointer in Basketball"
    response =  vectorDBAgent.app.invoke({"messages": [HumanMessage(content = question)]})
    print(response["messages"][-1].content)