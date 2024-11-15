import os
import urllib.parse
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

class SQLDB:
    def __init__(self, llm : any, verbose : bool = False) -> None:
        load_dotenv()
        self.db = SQLDatabase.from_uri(f"mysql+pymysql://{os.environ['SQL_USER']}:{urllib.parse.quote(os.environ['SQL_PASSWORD'])}@{os.environ['SQL_HOST']}/{os.environ['SQL_DATABASE_NAME']}",
                              sample_rows_in_table_info=3)
        
        self.toolkit = SQLDatabaseToolkit(db = self.db, llm = llm)
    
    
    def get_tools(self):
        return self.toolkit.get_tools()

if __name__ == '__main__':
    from langchain_mistralai import ChatMistralAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_core.messages import HumanMessage
    import os
    from dotenv import load_dotenv
    load_dotenv()
    rate_limiter = InMemoryRateLimiter(
        requests_per_second = 0.3,
        check_every_n_seconds = 0.1
    )
    sqlDB = SQLDB(
        ChatMistralAI(
            model_name = os.environ['MISTRAL_LLM_MODEL'], 
            temperature = 0.1, 
            rate_limiter = rate_limiter
            ), 
            verbose = True
    )
    print(sqlDB.db.run("select full_name from athletes where name like '%Pol Amat%';"))
