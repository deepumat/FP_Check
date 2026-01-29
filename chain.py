
""" LangGraph tool-calling, conversation-aware contract analysis agent

LLM is used ONLY for intent → tool planning All filtering & joins are deterministic """

=====================

Imports

=====================

from typing import List, Optional, Dict, Any from pydantic import BaseModel import pandas as pd

from langchain_openai import ChatOpenAI

Using a custom hosted LLM instead

import requests from langchain.prompts import ChatPromptTemplate from langchain.tools import tool from langgraph.graph import StateGraph, END from langgraph.prebuilt import ToolNode

=====================

Mock data loading (replace with CSVs)

=====================

contracts_df = pd.DataFrame([ {"file_id": "C1", "file_name": "contract_a.pdf", "vendor": "ABC"}, {"file_id": "C2", "file_name": "contract_b.pdf", "vendor": "XYZ"}, {"file_id": "C3", "file_name": "contract_c.pdf", "vendor": "ABC"}, ])

clauses_df = pd.DataFrame([ {"clause_id": "CL1", "clause_name": "termination"}, {"clause_id": "CL2", "clause_name": "indemnity"}, {"clause_id": "CL3", "clause_name": "confidentiality"}, ])

mapping_df = pd.DataFrame([ {"file_id": "C1", "clause_id": "CL1"}, {"file_id": "C1", "clause_id": "CL3"}, {"file_id": "C2", "clause_id": "CL1"}, {"file_id": "C2", "clause_id": "CL2"}, ])

=====================

Helper (deterministic)

=====================

def normalize_clause_via_api(raw_clause: str) -> str: """ External API call to clause-normalization model. This model maps user phrases like "clause2", "exit clause" to canonical clause names like "indemnity", "termination".

Replace the mock logic with a real HTTP call.
"""
# --- MOCK IMPLEMENTATION ---
mapping = {
    "clause2": "indemnity",
    "exit clause": "termination",
    "nda": "confidentiality",
}
return mapping.get(raw_clause.lower(), raw_clause.lower())

def filter_without_clause(df: pd.DataFrame, clause_name: str) -> pd.DataFrame: clause_row = clauses_df[clauses_df["clause_name"] == clause_name] if clause_row.empty: return df

clause_id = clause_row.iloc[0]["clause_id"]
with_clause = mapping_df[
    mapping_df["clause_id"] == clause_id
]["file_id"].unique()

return df[~df["file_id"].isin(with_clause)]

=====================

LangGraph State (memory)

=====================

class ContractQueryState(BaseModel): # conversation memory vendor: Optional[str] = None missing_clauses: List[str] = [] present_clauses: List[str] = []

# latest user input (LangGraph requires inputs to be part of state)
user_input: Optional[str] = None

# execution output
results: Optional[List[str]] = None

=====================

Tools (return STATE DELTAS)

=====================

@tool

def add_missing_clause(clause_name: str) -> Dict[str, Any]: """ Add a missing clause constraint. Calls external clause-normalization model BEFORE updating state. """ normalized = normalize_clause_via_api(clause_name) return {"missing_clauses": [normalized]}

@tool

def set_vendor(vendor: str) -> Dict[str, Any]: """Set vendor filter""" return {"vendor": vendor}

@tool

def execute_query(state: dict) -> Dict[str, Any]: """Execute deterministic contract filtering""" df = contracts_df.copy()

if state.get("vendor"):
    df = df[df["vendor"] == state["vendor"]]

for clause in state.get("missing_clauses", []):
    df = filter_without_clause(df, clause)

return {"results": df["file_name"].tolist()}

=====================

LLM (tool-calling only)

=====================

=====================

Custom LLM Client (hosted model)

=====================

class HostedLLM: """ Minimal wrapper for a hosted LLM (OpenAI-compatible schema). Used ONLY for intent → tool planning. """

def __init__(self, endpoint: str, api_key: str = None):
    self.endpoint = endpoint
    self.api_key = api_key

def invoke(self, prompt: str) -> str:
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    headers = {"Content-Type": "application/json"}
    if self.api_key:
        headers["Authorization"] = f"Bearer {self.api_key}"

    resp = requests.post(
        self.endpoint,
        json=payload,
        headers=headers,
        timeout=10
    )
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]

llm = HostedLLM( endpoint="https://your-hosted-llm-endpoint/v1/chat/completions", api_key="YOUR_API_KEY" )

intent_prompt = ChatPromptTemplate.from_messages([ ( "system", """ You are a contract query planner.

Rules:

Use tools to update state

Never answer directly

Never do filtering yourself


Available clauses:

termination

indemnity

confidentiality


IMPORTANT:

If user refers to clauses indirectly (e.g. clause2, exit clause, NDA) you MUST pass the raw clause text to the add_missing_clause tool.


Current state: {state} """ ), ("human", "{input}") ])

=====================

LangGraph Nodes

=====================

def intent_node(state: ContractQueryState): """LLM planning node (ONLY LLM CALL) LangGraph nodes must accept ONLY state. """ formatted_prompt = intent_prompt.format( input=state.user_input, state=state.dict() ) return llm.invoke(formatted_prompt) )

tools = [add_missing_clause, set_vendor, execute_query]

tool_node = ToolNode(tools)

=====================

Build Graph

=====================

builder = StateGraph(ContractQueryState)

builder.add_node("intent", intent_node) builder.add_node("tools", tool_node)

builder.set_entry_point("intent") builder.add_edge("intent", "tools") builder.add_edge("tools", END)

graph = builder.compile()

=====================

Example Conversation

=====================

if name == "main": state = ContractQueryState()

print("\n--- Turn 1 ---")
state = graph.invoke({
    "vendor": state.vendor,
    "missing_clauses": state.missing_clauses,
    "present_clauses": state.present_clauses,
    "results": state.results,
    "user_input": "Give me contracts without indemnity"
})
print(state)

print("\n--- Turn 2 ---")
state = graph.invoke({
    "vendor": state.vendor,
    "missing_clauses": state.missing_clauses,
    "present_clauses": state.present_clauses,
    "results": state.results,
    "user_input": "Give me contracts without indemnity"
})
print(state)

print("\n--- Turn 3 ---")
state = graph.invoke({
    "vendor": state.vendor,
    "missing_clauses": state.missing_clauses,
    "present_clauses": state.present_clauses,
    "results": state.results,
    "user_input": "Give me contracts without indemnity"
})
print(state)

print("\nFinal Result:")
print(state.results)