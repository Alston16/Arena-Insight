from typing import List, Dict, Union, Literal
from trulens.core import TruSession, Feedback, Select
from trulens.apps.custom import TruCustomApp, instrument
from trulens.providers.litellm import LiteLLM
from trulens.providers.huggingface import Huggingface
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
        huggingfaceProvider = Huggingface()
        litellmProvider = LiteLLM(model_engine = os.environ['LITELLM_PROVIDER'] + '/' + os.environ['LITELLM_MODEL'])
        self.feedbacks = []

        if component_tested == 'query_processor':
            instrument.methods(QueryProcessor, ["get_context", "processQuery"])

        elif component_tested == 'vector_db_agent':
            instrument.methods(VectorDBAgent,  ["get_context", "processQuery"])
        
        elif component_tested == 'sql_db_agent':
            instrument.methods(SQLDBAgent,  ["get_context", "processQuery"])
        
        elif component_tested == 'web_search_agent':
            instrument.methods(WebSearchAgent,  ["get_context", "processQuery"])
        
        else:
            raise ValueError(f"Invalid component_tested: '{component_tested}'. Must be one of query_processor, vector_db_agent, sql_db_agent or web_search_agent.")

        
        if use_context_relevance:
            f_context_relevance = (
                Feedback(litellmProvider.context_relevance, name="Context Relevance")
                .on_input()
                .on(Select.RecordCalls.get_context.rets)
                .aggregate(np.mean)
            )
            self.feedbacks.append(f_context_relevance)
        
        if use_groundedness:
            f_groundedness = (
                Feedback(
                    huggingfaceProvider.groundedness_measure_with_nli, name="Groundedness"
                )
                .on(Select.RecordCalls.get_context.rets)
                .on_output()
            )
            self.feedbacks.append(f_groundedness)
        
        if use_answer_relevance:
            f_answer_relevance = (
                Feedback(
                    litellmProvider.relevance, name="Answer Relevance"
                )
                .on_input_output()
            )
            self.feedbacks.append(f_answer_relevance)

        run_dashboard(self.session)
    
    def get_tru_app(self, app : QueryProcessor, version : str) -> TruCustomApp:
        return TruCustomApp(
            app,
            app_name = "Olympics",
            app_version = version,
            feedbacks = self.feedbacks
        )

    def evaluate(self, apps : Dict[str, Union[QueryProcessor, str]], queries : List[str]) -> None:
        for app in apps:
            tru_app = self.get_tru_app(app["app"], app["version"])
            print("Testing version", app["version"])
            with tru_app as recording:
                for query in tqdm(queries, desc = "Testing queries", unit = "queries"):
                    if self.component_tested == 'query_processor':
                        app["app"].processQuery(query, [])
                    else:
                        app["app"].processQuery(query)
                    time.sleep(5)

if __name__ == '__main__':
    from langchain_mistralai import ChatMistralAI
    from langchain_core.rate_limiters import InMemoryRateLimiter

    from test_suite import queries

    load_dotenv()

    rate_limiter = InMemoryRateLimiter(
        requests_per_second = 0.1,
        check_every_n_seconds = 0.1
    )

    # tester = TruLensTester()
    # llm = ChatMistralAI(model_name = os.environ['MISTRAL_LLM_MODEL'],temperature=0.1, rate_limiter = rate_limiter)
    # apps = [
    #     {
    #         "app" : QueryProcessor(llm, verbose = True),
    #         "version" : "base"
    #     }
    # ]
    tester = TruLensTester(component_tested='vector_db_agent')
    llm = ChatMistralAI(model_name = os.environ['MISTRAL_LLM_MODEL'],temperature=0.1, rate_limiter = rate_limiter)
    apps = [
        {
            "app" : VectorDBAgent(llm, verbose = True),
            "version" : "base"
        }
    ]
    tester.evaluate(apps, queries)
    