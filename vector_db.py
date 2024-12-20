from langchain_chroma import Chroma
from langchain.tools import StructuredTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

class VectorDB:
    def __init__(self, persist_directory : str, embedding_function : any, verbose : bool = False, use_semantic_filtering : bool = True) -> None:
        self.vectorstore = Chroma(persist_directory = persist_directory, embedding_function = embedding_function)
        self.retriever = self.vectorstore.as_retriever()
        self.verbose = verbose
        self.use_semantic_filtering = use_semantic_filtering
        if use_semantic_filtering:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def search(self, query : str) -> str:
        if self.verbose:
            print("---RETRIEVING CONTEXT---")
        docs = list(map(lambda x : x.page_content, self.retriever.invoke(query)))
        if self.use_semantic_filtering:
            docs = sum(map(self.splitter.split_text, docs), [])
            query_embedding = self.embedding_model.encode(query)
            chunk_embeddings = self.embedding_model.encode(docs)
            scores = util.semantic_search(query_embedding, chunk_embeddings, top_k = 3)

            docs = [docs[hit['corpus_id']] for hit in scores[0]]
        response = "\n\n".join(docs)
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