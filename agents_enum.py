from enum import Enum

class Agent(Enum):
    SQL_DB_AGENT = "sql_db_agent"
    VECTOR_DB_AGENT = "vector_db_agent"
    WEB_SEARCH_AGENT = "web_search_agent"
    GENERATE = "generate"