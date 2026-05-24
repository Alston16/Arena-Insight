from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

class VectorDB:
    _classifier = None
    _embedding_model = None

    def __init__(self, persist_directory : str, embedding_function : any, verbose : bool = False, use_semantic_filtering : bool = True, use_metadata_filtering = True) -> None:
        self.vectorstore = Chroma(persist_directory = persist_directory, embedding_function = embedding_function)
        if not use_metadata_filtering:
            self.retriever = self.vectorstore.as_retriever()
        self.verbose = verbose
        self.use_metadata_filtering = use_metadata_filtering
        self.use_semantic_filtering = use_semantic_filtering
        
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        if use_metadata_filtering:
            if VectorDB._classifier is None:
                VectorDB._classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
            self.classifier = VectorDB._classifier

            self.labels = ["Badminton Rules", "Basketball Rules", "Boxing Rules", "Breaking Rules", "Fencing Rules", 
            "Gymnastics Rules", "Handball Rules", "Judo Rules", "Table Tennis Rules", "Taekwondo Rules",
            "Weightlifting Rules", "Wrestling Rules", "Olympic News", "Olympic Term Definition", "Player Data"]
        if use_semantic_filtering:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
            if VectorDB._embedding_model is None:
                VectorDB._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            self.embedding_model = VectorDB._embedding_model

    def search(self, query : str) -> str:
        if self.verbose:
            print("---RETRIEVING CONTEXT---")
        if self.use_metadata_filtering:
            threshold = 0.15
            classifier_results = self.classifier(query, candidate_labels = self.labels)
            classifier_labels = classifier_results['labels']
            clssifier_scores = classifier_results['scores']
            query_labels = [classifier_labels[i] for i in range(len(clssifier_scores)) if clssifier_scores[i] > threshold]
            retriever = self.vectorstore.as_retriever(search_kwargs = {"filter" : {"data-type" : {"$in" : query_labels}}})
            docs = list(map(lambda x : x.page_content, retriever.invoke(query)))
        else:
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
    import torch
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    vectorDB = VectorDB(
        os.environ['VECTOR_DB_DIRECTORY'], 
        HuggingFaceEmbeddings(model_name=os.environ['EMBEDDINGS_MODEL'], model_kwargs={'device': device}), 
        verbose = True
        )
    question = 'How to score a 3 pointer in Basketball'
    print(vectorDB.search(question))