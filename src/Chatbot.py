from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import langsmith # Import required here for langsmith tracing
import os
from src.DbContext import DbContext
from src.VectorStore import VectorStore
from src.LinguisticsHelper import LinguisticsHelper
from datetime import date
from src.CacheService import CacheService
from src.ContextCacheObj import ContextCacheObj
import re

class Chatbot():

    _LLM_MODEL = "gemini-2.0-flash-001"
    _LANGSMITH_TRACING = "true"
    
    def __init__(self, vector_store: VectorStore, cache_service: CacheService):
        """
        Initialize chatbot service.

        args:
            vector_store (VectorStore): Vector store to leverage for RAG.
            cache_service (CacheService): Cache service to use for context management.
        """
        self._set_env_vars()

        self._llm = ChatGoogleGenerativeAI(model=Chatbot._LLM_MODEL)

        #self._qa_chain = prompt_template | llm
        self._vector_store = vector_store

        # Instantiate db context
        self._db_context = DbContext()

        self._linguistics_helper = LinguisticsHelper()
        self._cache_service = cache_service

    def invoke_llm(self,
                   question: str,
                   context_results_limit: int,
                   llm_memory_buffer_size: int,
                   context_similarity_threshold: int,
                   context_query_memory_limit: int,
                   conversation_id: str = None,
                   user_id: str = None):
        """
        Send message to LLM.

        args:
            question (str): Message.
            context_results_limit (int): Vector store number of results.
            llm_memory_buffer_size (int): Conversation memory buffer.
            context_similarity_threshold (int): Similarity search threshold.
            context_query_memory_limit (int): Degree of subject carryover.
            conversation_id (str): ID of ongoing conversation in applicable.
            user_id (str): Username of sender.
        
        returns (str, str): Answer, conversation ID,.
        """
        
        # Add conversation to db if new (otherwise check if it's actually in db)
        # (this method will raise an error if a conversation with the given id is not found)
        conversation_id, is_new = self._consolidate_conversation_id(conversation_id, user_id)

        # Build memory if applicable (if the exchange is part of an existing convo)
        memory = self._build_memory(conversation_id, llm_memory_buffer_size) if not is_new else []

        # Query vector store via similarity search
        context_query, used_prev_subjects = self._build_context_query(memory, question, context_query_memory_limit)
        context = self._build_context(context_query, context_results_limit, context_similarity_threshold)

        # Short circuit before LLM invocation if RAG context isn't sufficient
        if not context:
            return "I couldn't find enough relevant context to answer that question.", conversation_id
        
        # Create cache object for context
        contextCacheObj = ContextCacheObj(conversation_id+"_last_context", context)

        # Add context from previous question if convo not new (just for additional memory
        # capabilities)
        if not is_new:
            prev_context = self._cache_service.get(conversation_id+"_last_context")
            if prev_context:
                context += "\n" + prev_context.data
        
        # Add context (technically cache object) to cache
        self._cache_service.add(contextCacheObj)

        # Build prompt template and chain to LLM
        prompt_template = self._build_prompt_template(memory)
        qa_chain = prompt_template | self._llm

        # Invoke the model through the prompt chain
        response = qa_chain.invoke({
            "context": context,
            "question": question
        })

        full_answer = response.content

        # Extract structured info from final answer
        answer, subjects = self._extract_response(full_answer)

        # Add exchange to db (used for memory retrieval if more messages are asked in the conversation)
        self._add_exchange_to_db(conversation_id=conversation_id,
                                question=question,
                                answer=answer,
                                user_id=user_id,
                                subjects=subjects,
                                used_prev_subjects=used_prev_subjects)

        # Return the full answer string
        return answer, conversation_id
    
    def _build_memory(self, conversation_id: int, llm_memory_buffer_size: int):
        """
        Retrieve previous conversation exchanges to use for LLM invocation.

        args:
            conversation_id (int): ID of ongoing conversation.
        
        returns (str, str, str[]): AI/human, message, subjects.
        """
        # Get k most recent exchanges of conversation from db and format to be compatible with LangChain
        convo_history = self._db_context.get_recent_exchanges(conversation_id, llm_memory_buffer_size)
        memory = []
        for _, is_system, _, message, _, subjects in convo_history:
            memory.append(["ai" if is_system else "human", message, subjects])
        return memory

    
    def _consolidate_conversation_id(self, conversation_id: str = None, user_id: str = None):
        """
        Create conversation if it doesn't exist (if convo ID is null).

        args:
            conversation_id (int): ID of conversation if ongoing.
            user_id (str): Username.
        
        returns (str, bool): Conversation ID, whether conversation is new/ongoing.
        """
        # Create conversation if no conversation_id was passed in
        is_new = False
        if not conversation_id:
            conversation_id = self._db_context.create_conversation(user_id)
            is_new = True
        else:
            # Otherwise, check if conversation exists
            conversation = self._db_context.get_conversation_by_id(conversation_id)
            if not conversation:
                raise ValueError("Conversation not found")
        return conversation_id, is_new
    
    def _add_exchange_to_db(self, conversation_id: str, question: str, answer: str, user_id: str, subjects: list[str] = None, used_prev_subjects: bool = False):
        """
        Add AI-human exchange to database.

        args:
            conversation_id (int): ID of conversation.
            question (str): Human question.
            answer (str): AI answer.
            user_id (str): Username.
            subjects (str[]): Subjects of exchange.
            used_prev_subjects (bool): Whether previous exchange's subjects were used in
            similarity search.
        """
        # Add user and AI messages to db
        self._db_context.add_message(conversation_id, is_system=False, message=question, user_id=user_id) # user msg
        self._db_context.add_message(conversation_id, is_system=True, message=answer, subjects=subjects, used_prev_subjects=used_prev_subjects) # ai msg
    
    def _extract_response(self, response: str):
        """
        Unbox LLM response (required because LLM outputs both answer and inferred
        subjects in a specific format).

        args:
            response (str): LLM response.
        
        returns (str, str[]): answer, inferred subjects.
        """
        # Pattern to match the final underscore-wrapped subjects (e.g., _Slayer, Pantera_)
        match = re.search(r'_(.*?)_\s*$', response)
        if match:
            subjects_str = match.group(1)
            subjects = [s.strip() for s in subjects_str.split(',')]
            response = response[:match.start()].rstrip()
            return response, subjects
        else:
            # If no match is found, return the full text and an empty list
            return response, []
    
    def _build_context(self, question: str, context_results_limit: int, context_similarity_threshold: int):
        """
        Build context for LLM based on similarity search.

        args:
            question (str): Human question.
            context_results_limit (int): Vector store number of results.
            context_similarity_threshold (int): Similarity search threshold.
        
        returns (str): Built context.
        """
        # Run similarity search
        similar_chunks = self._vector_store.similarity_search(question, context_results_limit)

        # Filter by similarity score threshold
        filtered_chunks = [
            (text, url) for (text, url, score) in similar_chunks if score > context_similarity_threshold
        ]

        # Short-circuit if no good matches
        if not filtered_chunks:
            return None

        # Build context string for RAG prompt
        context = "\n".join([f"{text}\n({url})\n" for text, url in filtered_chunks])

        return context
    
    def _build_context_query(self, memory: list[tuple[str, str]], question: str, context_query_memory_limit: int):
        """
        Build vector store similarity search query (whether to carryover subjects
        if ongoing conversation).

        args:
            memory ((str, str)[]): Previous exchanges in conversation.
            question (str): Human question.
            context_query_memory_limit (int): Degree of subject carryover.
        
        returns (str): Built context query.
        """
        # Short circuit if no memory, meaning message must be start of new convo
        if not memory:
            return (question, False)

        # Get subjects of most recent "ai" message (system response)
        # (multiply context_query_memory_limit by 2 because 1 exchange is comprised of 2 messages)
        subjects = []
        for role, _, msg_subjs in memory[len(memory)-context_query_memory_limit*2:][::-1]:
            if role == "ai" and msg_subjs:
                subjects += msg_subjs

        # Check if subjects should be carried over based on conferential characteristics and entity overlap (see: LinguisticsHelper class)
        # (subjects list should almost always be populated at this point unless the LLM gave an "I don't know" response or the like)
        if subjects and self._linguistics_helper.should_carry_subjects(question, subjects):
            return (question + f" ({', '.join(subjects)})", True)
        
        # Return just the question if subjects shouldn't be carried (meaning context is different and message likely isn't a follow-up question)
        return (question, False)
            
    
    def _build_prompt_template(self, memory: list[(str, str)]):
        """
        Build prompt template for LLM invocation.

        args:
            memory ((str, str)[]): Previous exchanges in conversation.
        
        returns (str): Built prompt template.
        """
        # Create a structured prompt template with a system message and memory of previous exchanges
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self._get_sys_prompt())] + [tuple(msg[:2]) for msg in memory] + [("human", "{context}\n\nQuestion: {question}")]
        )

        return prompt_template
    
    def _get_formatted_date(self):
        """
        Get current date in mm/dd/YYYY format.

        args:
            memory ((str, str)[]): Previous exchanges in conversation.
        
        returns (str): Current, formatted date.
        """
        today = date.today()
        formatted = today.strftime("%-m/%-d/%Y")
        return formatted

    def _get_sys_prompt(self):
        """
        Get system prompt for LLM invocation.
        
        returns (str): System prompt.
        """
        # Get system prompt from llm_system_prompt.txt
        with open("src/prompts/llm_system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
            system_prompt = system_prompt.replace("{{date}}", self._get_formatted_date())
            return system_prompt
    
    def _set_env_vars(self):
        """
        Mount environment variables.
        """
        if Chatbot._LANGSMITH_TRACING:
            # LangSmith tracing
            os.environ["LANGSMITH_TRACING_V2"] = Chatbot._LANGSMITH_TRACING
            os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

        # langchain_google_genai (Gemini)
        os.environ["GOOGLE_API_KEY"] = os.getenv("LLM_API_KEY")

        # Suppress Tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"