import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy.engine import URL


DEFAULT_SQL_HOST = "mysql-17e5d930-arenainsight.c.aivencloud.com"
DEFAULT_SQL_PORT = 20787
DEFAULT_SQL_USER = "avnadmin"
DEFAULT_SQL_DATABASE = "olympics_data"


def _clean_env_value(value: str) -> str:
    return value.strip().strip("\"'")


def _get_sqlalchemy_url() -> URL:
    raw_host = _clean_env_value(os.getenv("SQL_HOST", DEFAULT_SQL_HOST))
    raw_port = os.getenv("SQL_PORT")

    host = raw_host
    port = _clean_env_value(raw_port) if raw_port else ""
    if not port and raw_host.count(":") == 1:
        candidate_host, candidate_port = raw_host.rsplit(":", 1)
        if candidate_port.isdigit():
            host = candidate_host
            port = candidate_port

    user = _clean_env_value(os.getenv("SQL_USER", DEFAULT_SQL_USER))
    password = _clean_env_value(os.getenv("SQL_PASSWORD", DEFAULT_SQL_PASSWORD))
    database = _clean_env_value(os.getenv("SQL_DATABASE_NAME", DEFAULT_SQL_DATABASE))

    return URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(port or DEFAULT_SQL_PORT),
        database=database,
    )

class SQLDB:
    def __init__(self, llm : any, verbose : bool = False) -> None:
        load_dotenv()
        print(_get_sqlalchemy_url())
        self.db = SQLDatabase.from_uri(
            _get_sqlalchemy_url(),
            sample_rows_in_table_info=3,
            engine_args={
                "pool_pre_ping": True,
                "connect_args": {
                    "charset": "utf8mb4",
                    "connect_timeout": 10,
                    "read_timeout": 10,
                    "write_timeout": 10,
                },
            },
        )
        
        self.toolkit = SQLDatabaseToolkit(db = self.db, llm = llm)
    
    
    def get_tools(self):
        return self.toolkit.get_tools()

if __name__ == '__main__':
    from llm_factory import create_llm

    sqlDB = SQLDB(
        create_llm(),
        verbose = True
    )
    print(sqlDB.db.run("select full_name from athletes where name like '%Pol Amat%';"))
