from flask import Flask, request, jsonify, Response
from src.Chatbot import Chatbot
from src.VectorStore import VectorStore
from src.CacheService import CacheService
from src.BacklinkRetriever import BacklinkRetriever
import os
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# SET PlugRAG CONFIGURATION
CONTEXT_RESULTS_LIMIT = 10 # Vector store number of results
LLM_MEMORY_BUFFER_SIZE = 3 # Conversation memory buffer
CONTEXT_SIMILARITY_THRESHOLD = 0 # Similarity search threshold
CONTEXT_QUERY_MEMORY_LIMIT = 1 # Must be less than or equal to LLM_MEMORY_BUFFER_SIZE (subject carry over basically)
TOKEN = os.getenv("SITE_TOKEN")

# Instantiate VectorStore, CacheService, 
vectorStore = VectorStore()
cache_service = CacheService()
chatbot = Chatbot(vectorStore, cache_service)
backlinkRetriever = BacklinkRetriever(vectorStore)

# Define Flask app
app = Flask(__name__)

@app.before_request
def authorize():
    """
    Define malware to authorize every request based on passed token. Short circuits
    request if token not provided or correct.
    
    returns (Response): JSON object effectuating 401 error if token invalid.
    """

    token = None

    if request.method == "POST":
        try:
            data = request.get_json(force=True, silent=True) or {}
            token = data.get('token')
        except Exception:
            token = None
    elif request.method == "GET":
        token = request.args.get('token')

    if not token or token != TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

@app.route('/chat', methods=['POST'])
def chat():
    """
    Invoke RAG system.

    args:
        user_id (str): User identifier.
        conversation_id (str): Unique converstion identifier to track conversation
                               memory and facilitate memory strategy.
        question (str): Question to pass through LLM.
    
    returns (Response): RAG system/LLM response in JSON format.
    """

    data = request.get_json()
    user_id = data.get('user_id')
    conversation_id = data.get('conversation_id')
    question = data.get('question')

    # Validate inputs
    if not all([question]):
        return jsonify({"error": "Missing one or more required fields"}), 400
    
    if (conversation_id and len(conversation_id) > 36) or len(question) > 200:
        return jsonify({"error": "Invalid field length"}), 400
    
    # Invoke LLM
    answer, conversation_id = chatbot.invoke_llm(
        question,
        CONTEXT_RESULTS_LIMIT,
        LLM_MEMORY_BUFFER_SIZE,
        CONTEXT_SIMILARITY_THRESHOLD,
        CONTEXT_QUERY_MEMORY_LIMIT,
        conversation_id,
        user_id
    )

    # Return JSON answer
    return jsonify({
        "answer": answer,
        "question": question,
        "conversation_id": conversation_id,
        "user_id": user_id
    })

@app.route('/update_document', methods=['POST'])
def update_document():
    """
    Add document or update it (if exists) in vector store.

    args:
        article_id (int): Article identifier.
        title (str): Title/headline of article.
        content (str): Article text.
        url_slug (str): URL slug of article.
        datetime (str): Publish date of article.
    
    returns (Response): JSON response denoting successful addition to vector store.
    """

    data = request.get_json()
    article_id = data.get('article_id')
    title = data.get('title')
    content = data.get('content')
    url_slug = data.get('url_slug')
    datetime = data.get('datetime')

    # Validate inputs
    if not all([article_id, title, content, url_slug, datetime]):
        return jsonify({"error": "Missing one or more required fields"}), 400
    
    # Add/update document in vector store
    vectorStore.update_in_vector_store(article_id,
                                       title,
                                       content,
                                       url_slug,
                                       datetime)
    
    # Return JSON response denoting success
    return jsonify({"article_id": article_id, "response": "successfully updated"})

@app.route('/delete_document', methods=['POST'])
def delete_document():
    """
    Remove document from vector store.

    args:
        article_id (int): Article identifier.
    
    returns (Response): JSON response denoting successful deletion from vector store.
    """

    data = request.get_json()
    article_id = data.get('article_id')

    # Validate input
    if not all ([article_id]):
        return jsonify({"error": "Missing one or more required fields"}), 400
    
    # Delete article from vector store
    vectorStore.delete_from_vector_store(article_id)

    # Return JSON response denoting success
    return jsonify({"article": article_id, "response": "successfully deleted"})

@app.route('/clean_cache', methods=['POST'])
def clean_cache():
    """
    Clean in-memory cache that facilitates model context strategy.
    
    returns (Response): JSON response denoting successful cache cleanup.
    """

    # Clean up in-memory cache
    cache_service.cleanup()

    # Return JSON response denoting success
    return jsonify({"response": "successfully cleaned cache"})

@app.route('/retrieve_backlinks', methods=['POST'])
def retrieve_backlinks():
    """
    Retrieve backlinks/related articles based on semantic (cosine) similarity score.

    args:
        article (str): Article text.
    
    returns (Response)): JSON response of nearest matches according to similarity score.
    """

    data = request.get_json()
    article = data.get('article')

    # Validate input
    if not all ([article]):
        return jsonify({"error": "Missing one or more required fields"}), 400
    
    # Retrieve backlinks from vector store based on similarity score
    response = backlinkRetriever.retrieve_backlinks(article)

    # Return JSON response of nearest matches
    return jsonify({"response": response})

if __name__ == '__main__':
    """
    Initialize Flask app locally on 0.0.0.0:5050.
    """
    app.run(host='0.0.0.0', port=5050)