from typing import List, Dict, Literal, Any
from trulens.core import TruSession, Feedback
from trulens.core.metric.selector import Selector
from trulens.apps.app import TruApp, instrument
from trulens.providers.litellm import LiteLLM
from trulens.dashboard.run import run_dashboard
import os
import time
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from query_processor import QueryProcessor
from vector_db_agent import VectorDBAgent
from sql_db_agent import SQLDBAgent
from web_search_agent import WebSearchAgent

class TruLensTester:
    def __init__(self, component_tested : Literal['query_processor', 'vector_db_agent', 'sql_db_agent', 'web_search_agent'] = 'query_processor', use_context_relevance : bool = True, use_groundedness : bool = True, use_answer_relevance : bool = True) -> None:
        load_dotenv()
        self.component_tested = component_tested
        self.session = TruSession()
        self.session.reset_database()
        self.litellmProvider = LiteLLM(model_engine = os.environ['LITELLM_PROVIDER'] + '/' + os.environ['LITELLM_MODEL'])
        self.feedbacks = []

        if component_tested == 'query_processor':
            QueryProcessor.get_context = instrument(QueryProcessor.get_context)
            QueryProcessor.processQuery = instrument(QueryProcessor.processQuery)

        elif component_tested == 'vector_db_agent':
            VectorDBAgent.get_context = instrument(VectorDBAgent.get_context)
            VectorDBAgent.processQuery = instrument(VectorDBAgent.processQuery)
        
        elif component_tested == 'sql_db_agent':
            SQLDBAgent.get_context = instrument(SQLDBAgent.get_context)
            SQLDBAgent.processQuery = instrument(SQLDBAgent.processQuery)
        
        elif component_tested == 'web_search_agent':
            WebSearchAgent.get_context = instrument(WebSearchAgent.get_context)
            WebSearchAgent.processQuery = instrument(WebSearchAgent.processQuery)
        
        else:
            raise ValueError(f"Invalid component_tested: '{component_tested}'. Must be one of query_processor, vector_db_agent, sql_db_agent or web_search_agent.")

        
        if use_context_relevance:
            # Select context from the stored _last_context attribute set by processQuery
            f_context_relevance = (
                Feedback(self.litellmProvider.context_relevance, name="Context Relevance")
                .on({
                    "question": Selector.select_record_input(),
                    "context": Selector(lambda rec: rec.app._last_context if hasattr(rec.app, '_last_context') else "")
                })
                .aggregate(np.mean)
            )
            self.feedbacks.append(f_context_relevance)
        
        if use_groundedness:
            # Select context from the stored _last_context attribute set by processQuery
            f_groundedness = (
                Feedback(
                    self.litellmProvider.groundedness_measure_with_cot_reasons, name="Groundedness"
                )
                .on({
                    "source": Selector(lambda rec: rec.app._last_context if hasattr(rec.app, '_last_context') else ""),
                    "statement": Selector.select_record_output()
                })
            )
            self.feedbacks.append(f_groundedness)
        
        if use_answer_relevance:
            f_answer_relevance = (
                Feedback(
                    self.litellmProvider.relevance, name="Answer Relevance"
                )
                .on_input_output()
            )
            self.feedbacks.append(f_answer_relevance)

        run_dashboard(self.session)
    
    def get_tru_app(self, app : Any, version : str) -> TruApp:
        return TruApp(
            app,
            app_name = "Olympics",
            app_version = version,
            feedbacks = self.feedbacks
        )

    def evaluate(self, apps : List[Dict[str, Any]], queries : List[str]) -> None:
        for app in apps:
            tru_app = self.get_tru_app(app["app"], app["version"])
            print("Testing version", app["version"])
            with tru_app:
                for query in tqdm(queries, desc = "Testing queries", unit = "queries"):
                    if self.component_tested == 'query_processor':
                        app["app"].processQuery(query, [])
                    else:
                        app["app"].processQuery(query)
                    time.sleep(5)

if __name__ == '__main__':
    from llm_factory import create_llm

    from test_suite import queries

    load_dotenv()

    # Test the application
    tester = TruLensTester()
    llm = create_llm()
    apps = [
        {
            "app" : QueryProcessor(llm),
            "version" : "base"
        }
    ]
    tester.evaluate(apps, queries)

    # Test Vector DB Agent
    # tester = TruLensTester(component_tested='vector_db_agent')
    # llm = create_llm()
    # apps = [
    #     {
    #         "app" : VectorDBAgent(llm, use_semantic_filtering = False, use_metadata_filtering = False),
    #         "version" : "base"
    #     },
    #     {
    #         "app" : VectorDBAgent(llm, use_metadata_filtering = False),
    #         "version" : "with_semantic_filtering"
    #     },
    #     {
    #         "app" : VectorDBAgent(llm, use_semantic_filtering = False),
    #         "version" : "with_metadata_filtering"
    #     },
    #     {
    #         "app" : VectorDBAgent(llm),
    #         "version" : "with_metadata_and_semantic_filtering"
    #     }
    # ]
    # tester.evaluate(apps, queries[0:50])

    # # Test SQL DB Agent
    # tester = TruLensTester(component_tested='sql_db_agent')
    # llm = create_llm()
    # apps = [
    #     {
    #         "app" : SQLDBAgent(llm),
    #         "version" : "with_few_shot"
    #     },
    #     {
    #         "app" : SQLDBAgent(llm, use_few_shot = False),
    #         "version" : "without_few_shot"
    #     }
    # ]
    # tester.evaluate(apps, queries[50:100])

    # # Test Web Search Agent
    # tester = TruLensTester(component_tested='web_search_agent')
    # llm = create_llm()
    # apps = [
    #     {
    #         "app" : WebSearchAgent(llm),
    #         "version" : "base"
    #     }
    # ]
    # tester.evaluate(apps, queries[100:120])
    