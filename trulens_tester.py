from typing import List, Dict, Union
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

class TruLensTester:
    def __init__(self, use_context_relevance : bool = True, use_groundedness : bool = True, use_answer_relevance : bool = True) -> None:
        load_dotenv()
        self.session = TruSession()
        self.session.reset_database()
        huggingfaceProvider = Huggingface()
        litellmProvider = LiteLLM(model_engine = os.environ['LITELLM_PROVIDER'] + '/' + os.environ['LITELLM_MODEL'])
        instrument.methods(QueryProcessor, ["get_context", "processQuery"])
        self.feedbacks = []

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
                    app["app"].processQuery(query, [])
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

    tester = TruLensTester()
    llm = ChatMistralAI(model_name = os.environ['MISTRAL_LLM_MODEL'],temperature=0.1, rate_limiter = rate_limiter)
    apps = [
        {
            "app" : QueryProcessor(llm, verbose = True),
            "version" : "base"
        }
    ]
    tester.evaluate(apps, queries)
    