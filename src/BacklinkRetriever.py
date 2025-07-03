from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import TokenTextSplitter
import langsmith # Import required here for langsmith tracing
import os
from src.VectorStore import VectorStore
from CoreferenceResolver import CoreferenceResolver
import re
from transformers import AutoTokenizer

class BacklinkRetriever():

    _LLM_MODEL = "gemini-2.0-flash-001"
    _LANGSMITH_TRACING = "true"

    def __init__(self, vector_store: VectorStore):
        """
        Initialize BacklinkRetriever.

        args:
            vector_store (VectorStore): Underlying vector store access.
        """

        self._set_env_vars()
        self._llm = ChatGoogleGenerativeAI(model=BacklinkRetriever._LLM_MODEL)
        self._vector_store = vector_store
        self._entity_backfiller = CoreferenceResolver()
        self._tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def retrieve_backlinks(self, article: str, score_threshold: float = 0.5):
        """
        Retrieves the nearest backlink matches of articles in the vector db
        (i.e., the most related articles).

        args:
            article (str): The article to run against the vector store.
            score_threshold (float): The threshold to apply to similarity score results.
        
        returns [(text, article_id, url, score)]: Article, ID, URL, and similarity score
        """

        # Backfill named entities (rewrite article to be as specific as possible
        # before chunking so vector store similarity search results are more accurate)
        # A.K.A Coreforence Resolution
        article = self._entity_backfiller.invoke_llm(article)
        article_chunks = self._chunk_article(article)

        # Perform similarity search on article against vector store
        all_matches = self._vector_store.similarity_search_batch(article_chunks,
                                                                 similarity_search_results_limit=5)
        
        # Apply threshold (filter out any results below)
        filtered_results = self._filter_results(all_matches, score_threshold)

        # Rank results by similarity score
        sorted_results = sorted(filtered_results, key=lambda x: x[-1], reverse=True)

        return sorted_results
    
    def _filter_results(self, all_matches, score_threshold):
        """
        Filters out results below the similarity score threshold and duplicates
        (duplicates may surface if multiple chunks in the vector store belong to
        the same article).

        args:
            all_matches (text, article_id, url, score): Found vector store matches.
            score_threshold (float): The threshold to apply to similarity score results.
        
        returns [(text, article_id, url, score)]: Article, ID, URL, and similarity score
        """

        filtered_matches = []
        article_ids = set()
        for chunk_matches in all_matches:
            for (text, article_id, url, score) in chunk_matches:
                if score >= score_threshold and article_id not in article_ids:
                    filtered_matches.append((self._extract_title(text), self._extract_date(text), url, score))
                    article_ids.add(article_id)
        return filtered_matches
    
    def _extract_title(self, text: str):
        """
        Extracts title from article text via Regex.

        args:
            text (str): Article text.
        
        returns (str): Title of article.
        """

        # Match everything up to the date in parentheses
        match = re.match(r'^(.*?)\s*\(\d{2}/\d{2}/\d{4}\)', text)
        if match:
            return match.group(1).strip()
        else:
            # fallback: get first line if date not found
            return text.split('\n')[0].strip()
        
    def _extract_date(self, text: str):
        """
        Extracts publish date from article text via Regex.

        args:
            text (str): Article text.
        
        returns (str): Publish date of article.
        """

        match = re.search(r'\((\d{2}/\d{2}/\d{4})\)', text)
        if match:
            return match.group(1)
        else:
            return None

    def _chunk_article(self, article: str):
        """
        Chunks article via chunking strategy.

        args:
            article (str): Article text.
        
        returns [(text, article_id, url, score)]: Article in chunks.
        """

        splitter = self._get_chunking_strategy()
        return splitter.split_text(article)
    
    def _get_chunking_strategy(self):
        """
        Defines and returns chunking strategy.
        
        returns (TokenTextSplitter): Chunking strategy.
        """

        return TokenTextSplitter(
            chunk_size=128,         # ~ 2-3 sentences for MiniLM
            chunk_overlap=20,       # overlap to keep context across chunks
            length_function=self._count_tokens
            # encoding_name can be left None since we're using a custom tokenizer/count function
        )
    
    def _count_tokens(self, text: str) -> int:
        """
        Return length of article text in terms of tokens.

        args:
            text (str): Article text.
        
        returns (int): Number of tokens.
        """

        return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    def _set_env_vars(self):
        """
        Mount environment variables.
        """
        if BacklinkRetriever._LANGSMITH_TRACING:
            # LangSmith tracing
            os.environ["LANGSMITH_TRACING_V2"] = BacklinkRetriever._LANGSMITH_TRACING
            os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

        # langchain_google_genai (Gemini)
        os.environ["GOOGLE_API_KEY"] = os.getenv("LLM_API_KEY")

        # Suppress Tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"