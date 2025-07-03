# PlugRAG

PlugRAG is a powerful, extensible framework for building Retrieval-Augmented Generation (RAG) applications. It combines a conversational AI with a vector store for efficient similarity searches and a backlink retriever to find related articles. This framework is designed to be a solid foundation for creating sophisticated, context-aware chatbots and other AI-powered tools.

-----

## 🚀 Features

  * **Conversational AI:** A chatbot that can answer questions based on the information stored in your vector database.
  * **Vector Store Integration:** Seamlessly integrates with Pinecone to provide fast and accurate similarity searches.
  * **Backlink Retriever:** A tool to find related articles and content within your vector store.
  * **Database Integration:** Uses a PostgreSQL database to store conversation history and other relevant data.
  * **RESTful API:** A clean and simple API for interacting with the framework's features.

-----

## 🛠️ Setup and Installation

To get started with PlugRAG, you'll need to set up a Pinecone vector store, a PostgreSQL database, and configure your environment variables.

### 1\. Create a Pinecone Vector Store

1.  **Sign up for a free Pinecone account** at [pinecone.io](https://www.pinecone.io/).
2.  **Create a new index.** You can do this from the Pinecone dashboard.
3.  **Get your Pinecone API key and index name.** You'll need these for your `.env` file.

### 2\. Create a PostgreSQL Database

1.  **Set up a PostgreSQL database.** You can use a cloud provider like AWS RDS, or run it locally.

2.  **Get your database credentials:** You'll need the database name, user, password, host, and port for your `.env` file.

3.  **Initialize the database tables.** The following SQL commands can be used to create the necessary tables:

    ```sql
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
    CREATE TABLE Convos (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id TEXT,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE Msgs (
        id SERIAL PRIMARY KEY,
        convo_id UUID REFERENCES Convos(id) ON DELETE CASCADE,
        is_system BOOLEAN NOT NULL,
        user_id TEXT,
        message TEXT NOT NULL,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        subjects TEXT[],
        used_prev_subjects BOOLEAN NOT NULL
    );
    ```

### 3\. Populate the .env File

Create a `.env` file in the root directory of the project and add the following environment variables:

```
DB_NAME="your_db_name"
DB_USER="your_db_user"
DB_PASSWORD="your_db_password"
DB_HOST="your_db_host"
DB_PORT="your_db_port"

PINECONE_API_KEY="your_pinecone_api_key"

LLM_API_KEY="your_llm_api_key"
LANGSMITH_API_KEY="your_langsmith_api_key"
SITE_TOKEN="your_site_token"
```

### 4\. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

-----

## ▶️ Usage

Once you have completed the setup and installation, you can start the Flask application and begin making requests to the API endpoints.

### Run the Application

To run the Flask application, execute the following command in your terminal:

```bash
python -m src.app
```

The application will start on `http://0.0.0.0:5050`.

### API Endpoints

Here are the available API endpoints:

#### `/chat`

This endpoint allows you to interact with the chatbot.

  * **Method:** `POST`

  * **Body:**

    ```json
    {
        "token": "your_site_token",
        "user_id": "user_id",
        "conversation_id": "conversation_id",
        "question": "Your question here"
    }
    ```

#### `/update_document`

This endpoint allows you to add a new document or update an existing one in the Pinecone vector store.

  * **Method:** `POST`

  * **Body:**

    ```json
    {
        "token": "your_site_token",
        "article_id": 123,
        "title": "Article Title",
        "content": "Article content here...",
        "url_slug": "article-title",
        "datetime": "YYYY-MM-DD HH:MM:SS"
    }
    ```

#### `/delete_document`

This endpoint allows you to remove a document from the vector store.

  * **Method:** `POST`

  * **Body:**

    ```json
    {
        "token": "your_site_token",
        "article_id": 123
    }
    ```

#### `/clean_cache`

This endpoint cleans the in-memory cache.

  * **Method:** `POST`

  * **Body:**

    ```json
    {
        "token": "your_site_token"
    }
    ```

#### `/retrieve_backlinks`

This endpoint retrieves backlinks and related articles based on semantic similarity.

  * **Method:** `POST`

  * **Body:**

    ```json
    {
        "token": "your_site_token",
        "article": "Article content here..."
    }
    ```

-----

## ⚙️ Configuration

You can configure the behavior of the PlugRAG framework by modifying the following variables in `src/app.py`:

  * `CONTEXT_RESULTS_LIMIT`: The number of results to retrieve from the vector store for context.
  * `LLM_MEMORY_BUFFER_SIZE`: The number of previous conversation exchanges to use as memory for the LLM.
  * `CONTEXT_SIMILARITY_THRESHOLD`: The similarity score threshold for filtering vector store results.
  * `CONTEXT_QUERY_MEMORY_LIMIT`: The degree of subject carryover from previous turns in a conversation. Must be less than or equal to `LLM_MEMORY_BUFFER_SIZE`.