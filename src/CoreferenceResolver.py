from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import langsmith # Import required here for langsmith tracing
import os
from src.VectorStore import VectorStore

class CoreferenceResolver():

    _LLM_MODEL = "gemini-2.0-flash-001"
    _LANGSMITH_TRACING = "true"

    def __init__(self):
        """
        Initialize coreference resolver.
        """
        self._set_env_vars()
        self._llm = ChatGoogleGenerativeAI(model=CoreferenceResolver._LLM_MODEL)

    def invoke_llm(self, article: str):
        """
        Invoke coreference resolver (rewrite text to eliminate coreferential dependencies).

        args:
            article (str): Text to alter.
        
        returns (string): Altered text.
        """
        # Set up QA chain
        prompt_template = self._build_prompt_template()
        qa_chain = prompt_template | self._llm

        # Invoke the model through the prompt chain
        response = qa_chain.invoke({ "article": article })

        return response.content
    
    def _build_prompt_template(self):
        """
        Construct prompt template.
        
        returns (ChatPromptTemplate): Constructed prompt template.
        """
        # Create a structured prompt template with a system message and memory of previous exchanges
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self._get_sys_prompt())] + [("human", "Article: {article}")]
        )

        return prompt_template
    
    def _get_sys_prompt(self):
        """
        Get system prompt.
        
        returns (str): System prompt.
        """
        # Get system prompt from coreforence_resolver_prompt.txt
        with open("src/prompts/coreference_resolver_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
            return system_prompt
    
    def _set_env_vars(self):
        """
        Set environment variables.
        """
        # Configure LangSmith tracing
        os.environ["LANGSMITH_TRACING_V2"] = CoreferenceResolver._LANGSMITH_TRACING
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

        # langchain_google_genai (Gemini)
        os.environ["GOOGLE_API_KEY"] = os.getenv("LLM_API_KEY")