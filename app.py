import streamlit as st
import os
import json
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import TokenTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI

# Load credentials from Streamlit secrets
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize Neo4j Driver
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load all completed documents in Neo4j
def get_all_documents(driver):
    query = "MATCH (d:Document {status: 'Completed'}) RETURN d.fileName AS fileName"
    with driver.session() as session:
        result = session.run(query)
        return [record["fileName"] for record in result]

# Initialize Graph-based Retriever
def initialize_neo4j_vector(driver):
    return Neo4jVector.from_existing_graph(
        embedding=None,
        index_name="document_index",
        retrieval_query="MATCH (n) WHERE n.text CONTAINS $query RETURN n",
        graph=driver,
        node_label="Document",
        text_node_properties=["text"]
    )

# Initialize Gemini LLM
def get_gemini_llm():
    return ChatOpenAI(openai_api_key=GEMINI_API_KEY, model="gemini-1.5")

# Retrieve relevant documents from Neo4j
def retrieve_documents(graph, query):
    retriever = initialize_neo4j_vector(graph)
    docs = retriever.invoke({"messages": [HumanMessage(content=query)]})
    return docs

# Process query with RAG-based chatbot
def process_query(question):
    driver = get_neo4j_driver()
    graph = driver.session()
    
    # Retrieve relevant docs
    docs = retrieve_documents(graph, question)
    
    # Generate response using Gemini LLM
    llm = get_gemini_llm()
    chat_prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages"), ("human", "User question: {input}")]
    )
    rag_chain = chat_prompt | llm

    ai_response = rag_chain.invoke({"messages": [HumanMessage(content=question)], "context": docs})
    
    driver.close()
    
    return ai_response.content

# Streamlit UI
st.set_page_config(page_title="GraphRAG Chatbot", layout="centered")
st.header("💬 GraphRAG Chatbot - Neo4j + Gemini")

# User input
user_input = st.text_input("Ask me anything about the documents:", "")

# Handle query
if user_input:
    with st.spinner("Processing..."):
        response = process_query(user_input)
        st.write("**Chatbot:**", response)
