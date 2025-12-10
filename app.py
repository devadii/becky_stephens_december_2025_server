import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
# ⚠️ WARNING: Your API keys are visible here. In production, use os.getenv()
os.environ["OPENAI_API_KEY"] = "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"
QDRANT_API_KEY = "1AA6g1qbzOfRGAto149kePIZdl6ElfIk1ojpsMF-ee2Eidf7VNqbwA"
QDRANT_ENDPOINT = "https://b2cf1251-d3f6-4a22-bd3e-578cf632d2f3.europe-west3-0.gcp.cloud.qdrant.io:6333"
COLLECTION_NAME = "becky_stephens_b"

# --- PROMPT TEMPLATES ---

CHAT_TEMPLATE = """
You are a helpful assistant. Give answer to the question strictly from the given context.

Answer should must contain two parts. First part should be brief introduction of question from the given context. Second part should be main answer to the question from the given context. This main answer should must be precise, detailed and in categories.

Answer should be in following JSON string format.
object = {{ "heading": category heading, "description": category description in bullet points in array format}}
Append each object in to array like following format
main answer = [object, object, ....]
So the final format is given as
output =  {{ "message_intro": first part, "message_explaination":  [object, object, ......], "error": false}}
Finally this end result shoud only be a pure json 

if the context does not contain the answer to the question then return the exact same json response given below:
{{
    "message_intro": "We can only provide answers from Sterling Road’s content directory. Check out the Categories listed below for the topics covered.", 
    "message_explaination": [], 
    "error": true
}}
 
Context : {context}
Question: {question}
"""

EXERCISES_TEMPLATE = """
Give the exercises that are actionable and practicable from the given Context. The exercises should be relevant to the given Question. Also provide content_source link from the context for each exercise that is nearest to the exercise in the context if exists.

These exercises should be in multiple short independent paragraphs of maximumt 3 lines and should must have following JSON format:
        
object = {{"description": exercises paragraph, "source": link to exercise}}
array of objects = [object, object, ....]
So the final format is given as:
{{"response":  [object, object, ......], "error": false, "status": "success"}}

Context : {context}
Question: {question}
"""

STORIES_TEMPLATE = """
Give the stories or past events from the given context. These stories or past events should be relevant to the given question.
        
if there are multiple stories or past events then these stories or past events should be in multiple independent paragraphs, if there is only one story or past event then it should be a a single independent paragraph.

These stories or past events should must have following JSON format:

object = {{"description": stories or past events paragraph, "source": link to the story of past event}}
array of objects = [object, object, ....]
So the final formate is given as:
{{"response":  [object, object, ......], "error": false, "status": "success"}}

if past event or story does not exists in the give context the retun followig json:
{{"response":  [{{"description": "No story found"}}], "error": false, "status": "success"}}

Context : {context}
Question: {question}
"""

LINKS_TEMPLATE = """
From the given context provide all content_source link, whos content is relvent to the given question and has answer for the given question. 
       
Return the response in following JSON format:

{{ "links": [ {{"title": "frist link's title", "link": "first link"}}, ... ] }}

Context : {context}
Question: {question}
"""

SUMMARY_TEMPLATE = """
Summarize the given context precisely in JSON format.

{{"summary": "generated summary"}}

Context : {context}
Question: {question}
"""

# --- DATABASE CONNECTION ---

def get_retriever(collection_name):
    """Initializes the Qdrant Hybrid Retriever."""
    print("Connecting to Vector Store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # FastEmbedSparse is required for Hybrid Search (Qdrant/bm25)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    # Using k=5 to get top 5 relevant chunks
    return qdrant.as_retriever(search_kwargs={"k": 5})

# --- CHAIN FACTORY ---

def create_chain(retriever, template_string):
    """Creates a RAG chain with JSON enforcement."""
    
    # 1. The Prompt
    prompt = ChatPromptTemplate.from_template(template_string)
    
    # 2. The Model (With JSON Mode Enabled)
    # model_kwargs={"response_format": ...} ensures GPT gives valid JSON
    model = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini",
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    # 3. The Output Parser (Handles string -> dict conversion automatically)
    parser = JsonOutputParser()

    # 4. The Chain (LCEL)
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

# Initialize Resources
try:
    retriever = get_retriever(COLLECTION_NAME)
    
    # Create chains for each endpoint
    chains = {
        "chat": create_chain(retriever, CHAT_TEMPLATE),
        "exercises": create_chain(retriever, EXERCISES_TEMPLATE),
        "stories": create_chain(retriever, STORIES_TEMPLATE),
        "links": create_chain(retriever, LINKS_TEMPLATE),
        "summary": create_chain(retriever, SUMMARY_TEMPLATE),
    }
    print("✅ System initialized successfully.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

def process_request(chain_key, query):
    """Helper to run chain and handle errors."""
    try:
        # invoke() returns a python dict because of JsonOutputParser
        response = chains[chain_key].invoke(query)
        return response
    except Exception as e:
        print(f"Error in {chain_key}: {e}")
        return {"error": True, "message": "Failed to generate response", "details": str(e)}

@app.route("/", methods=["GET"])
def confirmation():
    return "Server is running (RAG Mode)..."

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    
    ai_response = process_request("chat", query)
    
    # Construct final response wrapper
    return jsonify({
        "user": query, 
        "ai_message": ai_response
    })

@app.route("/chat/exercises", methods=["POST"])
def exercises_endpoint():
    data = request.get_json()
    query = data.get("query", "")
    # The prompt already enforces the structure, we just pass it through
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)