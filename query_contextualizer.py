from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from prompts import contextualize_q_system_prompt

class QueryContextualizer:
    def __init__(self, llm : any, verbose : bool = False) -> None:
        self.verbose = verbose

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chatHistory"),
                ("human", "{question}"),
            ]
        )
        self.chain = contextualize_q_prompt | llm | StrOutputParser()

    def contextualize(self, query : str, chatHistory : List[Dict]) -> str:
        if chatHistory:
            response = self.chain.invoke({"question" : query, "chatHistory" : chatHistory})
        else:
            response = query
        if self.verbose:
            print("Contextualized query :", response)
        return response

if __name__ == '__main__':
    from langchain_google_genai import GoogleGenerativeAI
    import os
    from dotenv import load_dotenv
    load_dotenv()
    queryContextualizer = QueryContextualizer(
        GoogleGenerativeAI(model = os.environ["GOOGLE_LLM_MODEL"], temperature=0.1),
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
    question = "In which years' Olympics has he participated in"
    print(queryContextualizer.contextualize(question, chatHistory))
    