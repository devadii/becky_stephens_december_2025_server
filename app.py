import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Removed FastEmbedSparse from imports
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not OPENAI_API_KEY or not QDRANT_API_KEY:
    raise ValueError("❌ API Keys not found. Make sure .env file is created.")

# --- PROMPT TEMPLATES ---

CHAT_TEMPLATE = """
You are a helpful assistant. Answer the question strictly based on the provided Context.

Your response must be a valid JSON object strictly adhering to the following structure:

{{
    "message_intro": "A brief introduction to the answer based on the context.",
    "message_explaination": [
        {{
            "heading": "Category Heading",
            "description": ["Bullet point 1", "Bullet point 2", "Bullet point 3"]
        }}
    ],
    "error": false
}}

Guidelines:
1. "message_intro": Brief summary.
2. "message_explaination": The main detailed answer categorized. The "description" must be an array of strings (bullet points).
3. If the Context does not contain the answer, return exactly:
{{
    "message_intro": "We can only provide answers from Sterling Road’s content directory. Check out the Categories listed below for the topics covered.", 
    "message_explaination": [], 
    "error": true
}}

Context: {context}
Question: {question}
"""

EXERCISES_TEMPLATE = """
Identify actionable and practicable exercises from the given Context relevant to the Question.

Output must be a valid JSON object strictly adhering to the following structure:

{{
    "response": [
        {{
            "description": "A short independent paragraph (max 3 lines) describing the exercise.",
            "source": "URL link to the content source (or null if not found)"
        }}
    ],
    "error": false,
    "status": "success"
}}

Context: {context}
Question: {question}
"""

STORIES_TEMPLATE = """
Identify stories or past events from the given Context relevant to the Question.

Output must be a valid JSON object strictly adhering to the following structure:

{{
    "response": [
        {{
            "description": "A standalone paragraph describing the story or event.",
            "source": "URL link to the story (or null if not found)"
        }}
    ],
    "error": false,
    "status": "success"
}}

If no stories are found in the context, return:
{{
    "response": [{{"description": "No story found"}}],
    "error": false,
    "status": "success"
}}

Context: {context}
Question: {question}
"""

LINKS_TEMPLATE = """
Identify all content source links from the Context that contain answers relevant to the Question.

Output must be a valid JSON object strictly adhering to the following structure:

{{
    "links": [
        {{
            "title": "Title of the link",
            "link": "The actual URL"
        }}
    ]
}}

Context: {context}
Question: {question}
"""

SUMMARY_TEMPLATE = """
Summarize the given context precisely.

Output must be a valid JSON object strictly adhering to the following structure:

{{
    "summary": "The generated summary text."
}}

Context: {context}
Question: {question}
"""

# --- DATABASE CONNECTION ---

def get_retriever(collection_name):
    """Initializes the Qdrant Dense Retriever (Standard Semantic Search)."""
    print("Connecting to Vector Store...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    
    # We removed FastEmbedSparse here.
    # We switch retrieval_mode to DENSE.
    
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.DENSE, # Changed from HYBRID to DENSE
    )
    
    # Using k=5 to get top 5 relevant chunks
    return qdrant.as_retriever(search_kwargs={"k": 5})

# --- CHAIN FACTORY ---

def create_chain(retriever, template_string):
    """Creates a RAG chain with JSON enforcement."""
    
    prompt = ChatPromptTemplate.from_template(template_string)
    
    model = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini",
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=OPENAI_API_KEY
    )
    
    parser = JsonOutputParser()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    return chain

# --- FLASK APPLICATION ---

app = Flask(__name__)
CORS(app)

try:
    retriever = get_retriever(COLLECTION_NAME)
    
    chains = {
        "chat": create_chain(retriever, CHAT_TEMPLATE),
        "exercises": create_chain(retriever, EXERCISES_TEMPLATE),
        "stories": create_chain(retriever, STORIES_TEMPLATE),
        "links": create_chain(retriever, LINKS_TEMPLATE),
        "summary": create_chain(retriever, SUMMARY_TEMPLATE),
    }
    print("✅ System initialized successfully (Dense Mode).")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

def process_request(chain_key, query):
    try:
        response = chains[chain_key].invoke(query)
        return response
    except Exception as e:
        print(f"Error in {chain_key}: {e}")
        return {"error": True, "message": "Failed to generate response", "details": str(e)}

@app.route("/", methods=["GET"])
def confirmation():
    return "Server is running (RAG Mode - Dense Only)..."

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    ai_response = process_request("chat", query)
    return jsonify({"user": query, "ai_message": ai_response})

@app.route("/chat/exercises", methods=["POST"])
def exercises_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    return jsonify(process_request("exercises", query))

@app.route("/chat/stories", methods=["POST"])
def stories_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    return jsonify(process_request("stories", query))

@app.route("/chat/links", methods=["POST"])
def links_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    return jsonify(process_request("links", query))

@app.route("/chat/sum", methods=["POST"])
def summary_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    return jsonify(process_request("summary", query))