"""
SQL Database Chat Application using Streamlit and LangChain

This application provides an interactive chat interface to query SQL databases
(MySQL and PostgreSQL) using natural language. It leverages LangChain's SQL agent
capabilities powered by Groq's LLM to translate user questions into SQL queries
and return human-readable responses.
"""

# ============================================================================
# IMPORTS SECTION
# ============================================================================

import streamlit as st

# LangChain SQL Agent - Creates an agent that can interact with SQL databases
from langchain_community.agent_toolkits import create_sql_agent

# SQLDatabase - LangChain's wrapper for SQL database connections
from langchain_community.utilities import SQLDatabase

# StreamlitCallbackHandler - Displays LangChain agent's thought process in Streamlit
# Used for: Showing intermediate steps and reasoning of the agent in the UI
from langchain_community.callbacks import StreamlitCallbackHandler

# SQLDatabaseToolkit - Collection of tools for SQL operations
# Used for: Providing the agent with tools to query schema, run queries, etc.
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# SQLAlchemy - SQL toolkit and Object-Relational Mapping (ORM) library
# Used for: Creating database engines and managing database connections
from sqlalchemy import create_engine

# ChatGroq - LangChain integration for Groq's LLM API
# Used for: Accessing Groq's language models (llama-3.3-70b-versatile) for natural language understanding
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Langchain: Chat with your SQL Database", layout="wide")
st.title("Langchain: Chat with your SQL Database")

INJECTION_WARNING = """
SQL agent can be vulnerable to prompt injection attacks. Use a DB role that has limited permissions. Consider using a read-only role.
"""

# ============================================================================
# DATABASE CONFIGURATION - SIDEBAR INPUTS
# ============================================================================

# Define supported database types
database_type_radio_options = ["mysql","postgresql"]

# Radio button to select database type (MySQL or PostgreSQL)
selected_database_type = st.sidebar.radio(
    "Select Database Type", database_type_radio_options
)

# Database connection parameters
# These inputs allow users to configure their database connection
host = st.sidebar.text_input("Host", value="localhost")
port = st.sidebar.text_input("Port", value="3306")  # Default MySQL port
user = st.sidebar.text_input("User", value="root")
password = st.sidebar.text_input("Password", type="password", value="password")
database = st.sidebar.text_input("Database", value="poornah-capital")

# Groq API key for accessing the LLM
# This key is required to use Groq's language models
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=groq_api_key)

# ============================================================================
# INPUT VALIDATION
# ============================================================================

# Validate that Groq API key is provided
# Without this, the LLM cannot be initialized
if not groq_api_key:
  st.error("Please enter a Groq API Key")
  st.stop()  # Stop execution if API key is missing

# Validate that all database credentials are provided
# All fields are required to establish a database connection
if not host or not port or not database or not user or not password:
  st.error("Please enter all the database credentials")
  st.stop()  # Stop execution if any credential is missing


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

# Initialize the ChatGroq LLM with streaming enabled
# Model: llama-3.3-70b-versatile - A powerful language model for understanding and generating text
# Streaming: True - Enables real-time response streaming for better UX
llm=ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", streaming=True)


# ============================================================================
# DATABASE CONNECTION FUNCTION
# ============================================================================

@st.cache_resource(ttl="2h")  # Cache the database connection for 2 hours to improve performance
def configure_db(host, port, database, user, password):
  """
  Configure and return a SQLDatabase connection based on the selected database type.
  
  This function creates a database connection string, initializes a SQLAlchemy engine,
  and wraps it in LangChain's SQLDatabase utility for agent interaction.
  
  Args:
      host (str): Database host address
      port (str): Database port number
      database (str): Database name
      user (str): Database username
      password (str): Database password
  
  Returns:
      SQLDatabase: LangChain SQLDatabase object or None if database type is invalid
  """
  if selected_database_type == "postgresql":
    # PostgreSQL connection string format
    # Uses psycopg2 driver for PostgreSQL connectivity
    db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(db_url)
    return SQLDatabase(engine)
  elif selected_database_type == "mysql":
    # MySQL connection string format
    # Uses pymysql driver for MySQL connectivity
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(db_url)
    return SQLDatabase(engine)
  else:
    st.error("Please select a database type")
    return None


# ============================================================================
# DATABASE CONNECTION INITIALIZATION
# ============================================================================

# Create the database connection using the provided credentials
db = configure_db(host, port, database, user, password)

# Validate that database connection was successful
if not db:
  st.error("Please enter all the database credentials")
  st.stop()  # Stop execution if database connection failed


# ============================================================================
# SQL AGENT SETUP
# ============================================================================

# Create a toolkit with SQL database tools
# This provides the agent with capabilities to query schema, execute SQL, etc.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent with the following configuration:
# - llm: The language model to use for understanding queries
# - toolkit: Tools for interacting with the database
# - verbose: Show detailed agent reasoning (useful for debugging)
# - handle_parsing_errors: Gracefully handle errors in parsing agent output
# - agent: "zero-shot-react-description" - Agent type that reasons about actions without examples
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,
    agent="zero-shot-react-description"
) 


# ============================================================================
# CHAT INTERFACE - SESSION STATE MANAGEMENT
# ============================================================================

# Initialize chat history in session state or clear it when button is clicked
# Session state persists data across Streamlit reruns
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
  st.session_state.messages = []  # Initialize empty message list
  # Add initial greeting message from the assistant
  st.session_state.messages.append({"role": "assistant", "content": "Hi, I'm your SQL assistant. How can I help you?"})

# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================

# Display all previous messages in the chat interface
# Iterates through message history and renders each message with appropriate role
for message in st.session_state.messages:
  with st.chat_message(message["role"]):  # Creates a chat message container with role-based styling
    st.markdown(message["content"])  # Display message content

# ============================================================================
# HANDLE NEW USER INPUT
# ============================================================================

# Chat input widget - captures user's question about the database
# The walrus operator (:=) assigns the input to 'prompt' and checks if it's not empty
if prompt := st.chat_input("Ask a question about your database"):
  # Add user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})
  
  # Display user message in the chat interface
  with st.chat_message("user"):
    st.markdown(prompt)
  
  # Generate and display assistant response
  with st.chat_message("assistant"):
    # Create a placeholder for the "Thinking..." message
    message_placeholder = st.empty()
    message_placeholder.markdown("Thinking...")
    
    try:
      # Run the agent with the user's prompt
      # StreamlitCallbackHandler displays the agent's reasoning process in real-time
      response = agent.run(prompt, callbacks=[StreamlitCallbackHandler(st.container())])
      
      # Add assistant response to chat history
      st.session_state.messages.append({"role": "assistant", "content": response})
      
      # Display the final response
      st.write(response)
    except Exception as e:
      # Handle any errors that occur during agent execution
      # Add error message to chat history for user reference
      st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
      st.error(f"Error: {e}")  # Display error in red
  