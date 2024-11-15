from langchain_chroma import Chroma
from langchain.tools import StructuredTool

class VectorDB:
    def __init__(self, persist_directory : str, embedding_function : any, verbose : bool = False) -> None:
        self.vectorstore = Chroma(persist_directory = persist_directory, embedding_function = embedding_function)
        self.retriever = self.vectorstore.as_retriever()
        self.verbose = verbose

    def search(self, query : str) -> str:
        if self.verbose:
            print("---RETRIEVING CONTEXT---")
        docs = self.retriever.invoke(query)
        response = "\n\n".join(doc.page_content for doc in docs)
        if self.verbose:
            print("Vector Database Response :", response)
        return response
    
    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func = self.search,
            name = "search_vector_db",
            description = "Search a vector database created by adding information about various aspects of Olympics for relevant documents"
        )

if __name__ == '__main__':
    from langchain_huggingface import HuggingFaceEmbeddings
    from dotenv import load_dotenv
    import os
    load_dotenv()
    vectorDB = VectorDB(
        os.environ['VECTOR_DB_DIRECTORY'], 
        HuggingFaceEmbeddings(model_name=os.environ['EMBEDDINGS_MODEL']), 
        verbose = True
        )
    question = 'Tell me about Michael Phelps'
    print(vectorDB.search(question))