import streamlit as st
import google.generativeai as genai
from neo4j import GraphDatabase

# Load API Keys and Neo4j Credentials from Streamlit Secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

# Ensure API Key is Set
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it in Streamlit Secrets.")
    st.stop()

# Initialize Gemini 1.5 LLM
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Establish Neo4j AuraDB Connection
def get_neo4j_connection():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

# Query Neo4j for GraphRAG-based Knowledge
def query_neo4j(user_query):
    with get_neo4j_connection().session() as session:
        query = """
        MATCH (c:Chunk)-[:SOURCE]->(doc:Document)
        WHERE c.text CONTAINS $user_query OR toLower(c.text) CONTAINS toLower($user_query)
        OPTIONAL MATCH (c)-[r]->(related)
        RETURN DISTINCT c.text AS chunk, 
                        type(r) AS relationship, 
                        related.text AS related_chunk, 
                        doc.name AS source
        LIMIT 5
        """
        result = session.run(query, {"user_query": user_query})
        return [record.values() for record in result]

# Generate AI-Powered Response
def generate_chat_response(user_query):
    graph_data = query_neo4j(user_query)

    # If no relevant data is found
    if not graph_data:
        return "No relevant information was found in Neo4j AuraDB.", []

    # Extract graph-based knowledge
    structured_knowledge = "\n".join(
        [f"- {chunk[:300]}..." for chunk, _, _, _ in graph_data if chunk]
    )

    # Prepare detailed structured knowledge for "Show Details"
    detailed_info = [
        f"""
        <div style="font-size:14px; padding:10px; border-bottom: 1px solid #ddd;">
        <b>Chunk:</b> {chunk[:300]}...<br>
        <b>Relationship:</b> {relationship if relationship else "N/A"} → {related_chunk if related_chunk else "N/A"}<br>
        <b>Source Document:</b> {source if source else "Unknown"}
        </div>
        """
        for chunk, relationship, related_chunk, source in graph_data
    ]

    # Gemini 1.5 Prompt (GraphRAG Enhanced Answer)
    prompt = f"""
    You are an AI assistant using GraphRAG and Neo4j AuraDB.
    Below is structured knowledge extracted from the database:

    {structured_knowledge}

    Question: {user_query}
    Provide a well-reasoned, professional response using relevant knowledge.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response else "No relevant information was found.", detailed_info
    except Exception as e:
        return f"An error occurred while retrieving data: {str(e)}", []

# Streamlit UI
st.title("GraphRAG-Powered Chatbot (Gemini 1.5 & Neo4j AuraDB)")
st.write("Ask a question related to your knowledge graph.")

user_input = st.text_input("Enter your question:")

if user_input:
    response, detailed_info = generate_chat_response(user_input)
    st.markdown(f"**Chatbot Response:**\n\n{response}")

    # Show details button
    if detailed_info:
        if st.button("Show Details"):
            for detail in detailed_info:
                st.markdown(f"<p style='font-size:14px;'>{detail}</p>", unsafe_allow_html=True)
