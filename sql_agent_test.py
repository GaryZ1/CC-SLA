# -*- coding: utf-8 -*-


import os

os.environ["COHERE_API_KEY"] = "wDpHETIQ98RwVcWpi2irjvn6lOyEFioGqtrqmlZ0"

from langchain.agents import AgentExecutor
from langchain_cohere import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_cohere import CohereEmbeddings
from datetime import datetime, timedelta
import os
import json
import sqlite3
import os

from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

def get_engine_for_manufacturing_db():
    """Create an in-memory database with the manufacturing data tables."""
    connection = sqlite3.connect(":memory:", check_same_thread=False)

    # Read and execute the SQL scripts
    for sql_file in ['product_tracking.sql', 'status.sql']:
        with open(sql_file, 'r') as file:
            sql_script = file.read()
            connection.executescript(sql_script)

    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

# Create the engine
engine = get_engine_for_manufacturing_db()

# Create the SQLDatabase instance
db = SQLDatabase(engine)


MODEL="command-r-plus-08-2024"
llm = ChatCohere(model=MODEL,
                 temperature=0.1,
                 verbose=True,
                 cohere_api_key=os.getenv("COHERE_API_KEY"))


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
context = toolkit.get_context()
tools = toolkit.get_tools()

# print('**List of pre-defined Langchain Tools**')
# print([tool.name for tool in tools])


examples = [
    {
        "input": "What was the average processing time for all stations on April 3rd 2024?",
        "query": "SELECT station_name, AVG(CAST(duration AS INTEGER)) AS avg_processing_time FROM product_tracking WHERE date = '2024-04-03' AND zone = 'wip' GROUP BY station_name ORDER BY station_name;",
    },
    {
        "input": "What was the average processing time for all stations on April 3rd 2024 between 4pm and 6pm?",
        "query": "SELECT station_name, AVG(CAST(duration AS INTEGER)) AS avg_processing_time FROM product_tracking WHERE date = '2024-04-03' AND CAST(hour AS INTEGER) BETWEEN 16 AND 18 AND zone = 'wip' GROUP BY station_name ORDER BY station_name;",
    },
    {
        "input": "What was the average processing time for stn4 on April 3rd 2024?",
        "query": "SELECT AVG(CAST(duration AS INTEGER)) AS avg_processing_time FROM product_tracking WHERE date = '2024-04-03' AND station_name = 'stn4' AND zone = 'wip';",
    },
    {
        "input": "How much downtime did stn2 have on April 3rd 2024?",
        "query": "SELECT COUNT(*) AS downtime_count FROM status WHERE date = '2024-04-03' AND station_name = 'stn2' AND station_status = 'downtime';",
    },
    {
        "input": "What were the productive time and downtime numbers for all stations on April 3rd 2024?",
        "query": "SELECT station_name, station_status, COUNT(*) as total_time FROM status WHERE date = '2024-04-03' GROUP BY station_name, station_status;",
    },
    {
        "input": "What was the bottleneck station on April 3rd 2024?",
        "query": "SELECT station_name, AVG(CAST(duration AS INTEGER)) AS avg_processing_time FROM product_tracking WHERE date = '2024-04-03' AND zone = 'wip' GROUP BY station_name ORDER BY avg_processing_time DESC LIMIT 1;",
    },
    {
        "input": "Which percentage of the time was stn5 down in the last week of May?",
        "query": "SELECT SUM(CASE WHEN station_status = 'downtime' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS percentage_downtime FROM status WHERE station_name = 'stn5' AND date >= '2024-05-25' AND date <= '2024-05-31';",
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"),
                     model="embed-english-v3.0"),
    FAISS,
    k=5,
    input_keys=["input"],
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

system_prefix = """You are an agent designed to interact with a SQL database.
You are an expert at answering questions about manufacturing data.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Always start with checking the schema of the available tables.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

The current date is {date}.

For questions regarding productive time, downtime, productive or productivity, use minutes as units.

For questions regarding productive time, downtime, productive or productivity use the status table.

For questions regarding processing time and average processing time, use minutes as units.

For questions regarding bottlenecks, processing time and average processing time use the product_tracking table.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k","date"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Example formatted prompt
prompt_val = full_prompt.invoke(
    {
        "input": "What was the productive time for all stations today?",
        "top_k": 5,
        "dialect": "SQLite",
        "date":datetime.now(),
        "agent_scratchpad": [],
    }
)
# print(prompt_val.to_string())



agent = create_sql_agent(
   llm=llm,
   toolkit=toolkit,
   prompt=full_prompt,
   verbose=True
)



output=agent.invoke({
   "input": "Which station had the highest total duration in the wait zone?",
    "date": datetime.now()
})
print(output['output'])

