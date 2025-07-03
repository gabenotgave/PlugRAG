from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import re
import os

class VectorStore():

    def __init__(self):
        """
        Initialize VectorStore.
        """
        # Embeddings model
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        index_name = "articles"
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self._index = pc.Index(index_name)

    
    def similarity_search(self, text: str, similarity_search_results_limit: int):
        """
        Perform similarity search against vector store using specified text.

        args:
            text (str): Text to use for similarity search.
            similarity_search_results_limit (int): Top results to retrieve from
            search.
        
        returns: Top similarity_search_results_limit results.
        """
        question_vector = self._embedding_model.encode(text).tolist()

        results = self._index.query(
            vector=question_vector,
            top_k=similarity_search_results_limit,
            include_metadata=True
        )

        # Return list of (chunk_text, url, score)
        return [
            (
                match["metadata"].get("text", ""),
                match["metadata"].get("url", ""),
                match["score"]
            )
            for match in results.get("matches", [])
        ]

    def similarity_search_batch(self, texts: list[str], similarity_search_results_limit: int):
        """
        Perform similarity search against vector store using specified batches of text.

        args:
            text (str): Text to use for similarity search.
            similarity_search_results_limit (int): Top results to retrieve from
            search.
        
        returns: Top similarity_search_results_limit results for each batch.
        """
        # Batch encode all input texts (assuming your embedding model supports batch)
        question_vectors = self._embedding_model.encode(texts)  # shape: [batch_size, embedding_dim]

        all_results = []

        for vector in question_vectors:
            # Convert to list if needed by your index client
            vec_list = vector.tolist() if hasattr(vector, "tolist") else vector

            results = self._index.query(
                vector=vec_list,
                top_k=similarity_search_results_limit,
                include_metadata=True
            )

            # Parse results to desired output format
            batch_result = [
                (
                    match["metadata"].get("text", ""),
                    match["metadata"].get("article_id"),
                    match["metadata"].get("url", ""),
                    match["score"]
                )
                for match in results.get("matches", [])
            ]

            all_results.append(batch_result)

        # Returns a list of results per input text: List[List[(text, url, score)]]
        return all_results

        
    def add_to_vector_store(self, article_id: int, title: str, content: str, url_slug: str, datetime: str):
        """
        Add article to vector store.

        args:
            article_id (int): Article ID.
            title (str): Article title.
            content (str): Article text.
            url_slug (str): Article URL slug.
            datetime (str): Article published date.
        """
        text_splitter = self._get_chunking_strategy()

        date = "/".join(datetime.split()[0].split("-")[1:]) + f"/{datetime[:4]}"
        url = f"{article_id}/{url_slug}"
        full_text = f"{title.strip()} ({date})\n{self._process_content(str(content).strip())}"
        
        records = []

        chunks = text_splitter.split_text(full_text)
        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue
            
            embedding = self._embedding_model.encode(chunk).tolist()
            
            record = {
                "id": f"{article_id}_{idx}",
                "values": embedding,
                "metadata": {
                    "article_id": article_id,
                    "url": url,
                    "text": chunk
                }
            }
            records.append(record)
        
        self._upload_batches(records)

    def update_in_vector_store(self, article_id: int, title: str, content: str, url_slug: str, datetime: str):
        """
        Update article in vector store by removing it (if it exists) and
        performing add operation again.

        args:
            article_id (int): Article ID.
            title (str): Article title.
            content (str): Article text.
            url_slug (str): Article URL slug.
            datetime (str): Article published date.
        """
        # Delete old documents/chunks from vector store and add again with updated info
        self.delete_from_vector_store(article_id)
        self.add_to_vector_store(article_id, title, content, url_slug, datetime)

    def delete_from_vector_store(self, article_id: int):
        """
        Delete article from vector store based in article ID.

        args:
            article_id (int): ID of article to delete from vector store.
        """
        results = self._index.query(
            vector=[0.0]*384,  # dummy vector, we only care about filtering
            top_k=1000,
            filter={"article_id": {"$eq": article_id}},  # your target article_id
            include_metadata=False
        )

        ids_to_delete = [match["id"] for match in results.get("matches", [])]

        if ids_to_delete:
            self._index.delete(ids=ids_to_delete)

    def _get_chunking_strategy(self):
        """
        Get chunking strategy.
        
        returns (RecursiveCharacterTextSplitter): Chunking strategy to apply for additions
        to vector store.
        """
        # Define text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            #chunk_size=4096,       # ~2–3 sentences -- SUSPENDED because need to figure out
                                    # how to add title and date on top of each chunk instead
                                    # of once for the entire article.
            chunk_size=100000,
            chunk_overlap=200,     # maintain some context
            separators=["\n\n", "\n", ".", " "],  # prefer paragraph and sentence breaks
            length_function=len   # still based on character count
        )

        return text_splitter
    
    def _upload_batches(self, records, batch_size=100):
        """
        Add records to vector store in batches.

        args:
            records (list): Records to upload to vector store.
            batch_size (int): Number of records to upsert simultaneously.
        """
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self._index.upsert(batch)
    
    def _process_content(self, content):
        """
        Process content before adding to vector store.

        args:
            content (str): Text to process.
        
        returns (str): Processed text (stripped of HTML, embeds, and whitespace).
        """
        REMOVE_EMBEDS_R = re.compile('<(script)\b[^>]*>.*?</\1>')
        STRIP_HTML_R = re.compile('<.*?>')
        REMOVE_SPACE_BEFORE_PUNC_R = re.compile('\s+(?=[.,])')
        
        def remove_embeds(raw_html):
            result = re.sub(REMOVE_EMBEDS_R, "", raw_html)
            return result
        
        def strip_html(raw_html):
            result = re.sub(STRIP_HTML_R, ' ', raw_html)
            return result
        
        def remove_whitespace(text):
            result = '\n'.join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
            return result
        
        def remove_space_before_punctuation(text):
            result = re.sub(REMOVE_SPACE_BEFORE_PUNC_R, '', text)
            return result

        content = remove_embeds(content)
        content = strip_html(content)
        content = remove_whitespace(content)
        content = remove_space_before_punctuation(content)
        
        return content