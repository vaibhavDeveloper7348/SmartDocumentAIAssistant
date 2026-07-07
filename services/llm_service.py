"""
LLM Service

Purpose:
    Provides an interface to the Language Model used for generating answers.
    Supports multiple LLM backends with automatic fallback.

Why multiple LLM options?
    - OpenAI: High quality, requires API key (paid)
    - Ollama: Local, free, requires setup
    - HuggingFace: Local, free, works out-of-the-box

    Default is HuggingFace (free, no setup required) so the project
    works immediately without any API keys.

Interview Note:
    Understanding LLM options is important. Interviewers ask:
    - Why use HuggingFace models? (free, local, private)
    - Trade-offs between hosted vs local LLMs? (cost vs control)
    - What is the role of the LLM in RAG? (synthesize retrieved info)
    - How do you handle LLM failures? (fallback strategy)
"""

from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from app.config import Config


class LLMService:
    """
    Manages LLM initialization and provides a unified interface
    for generating responses.

    Automatically selects the best available LLM:
    1. OpenAI (if API key is configured)
    2. Ollama (if available)
    3. HuggingFace (default, always works)

    Each option has different trade-offs between quality, speed,
    and cost that you should discuss in interviews.
    """

    def __init__(self):
        """Initialize the LLM based on available configuration."""
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """
        Select and initialize the best available LLM.

        Choice priority:
        1. OpenAI - if API key is set (highest quality)
        2. Ollama - if configured (good quality, free, local)
        3. HuggingFace - default fallback (free, works everywhere)

        Returns:
            LangChain LLM instance

        Why this priority?
            OpenAI produces the best answers but costs money.
            Ollama is a good free alternative but requires setup.
            HuggingFace always works but is slower and less capable.
            This layered approach ensures the project always works.
        """
        # Priority 1: OpenAI (best quality, requires paid API key)
        if Config.OPENAI_API_KEY:
            print("Using OpenAI LLM")
            return ChatOpenAI(
                model=Config.OPENAI_MODEL_NAME,
                temperature=0.2,  # Low temperature for factual answers
                openai_api_key=Config.OPENAI_API_KEY,
            )

        # Priority 2: Ollama (good quality, free, local)
        # Requires Ollama installed and a model pulled
        try:
            import requests as req
            # Check if Ollama is running AND has the configured model
            resp = req.get(
                f"{Config.OLLAMA_BASE_URL}/api/tags",
                timeout=3,
            )
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Use configured model if available, otherwise use first available
                if Config.OLLAMA_MODEL_NAME in model_names:
                    selected_model = Config.OLLAMA_MODEL_NAME
                elif model_names:
                    selected_model = model_names[0]
                else:
                    raise Exception("No models found in Ollama")
                ollama_llm = Ollama(
                    model=selected_model,
                    base_url=Config.OLLAMA_BASE_URL,
                    temperature=0.2,
                )
                print(f"Using Ollama LLM ({selected_model})")
                return ollama_llm
        except Exception:
            pass  # Ollama not available, fall through

        # Priority 3: HuggingFace (always works, free, local)
        print("Using HuggingFace LLM (free, local)")
        # Use a small model that works on CPU
        hf_pipeline = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",
            max_new_tokens=200,
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM based on the given prompt.

        Args:
            prompt (str): Complete prompt with context and question

        Returns:
            str: Generated answer text

        How it works:
            1. The prompt contains retrieved chunks + the user's question
            2. The LLM reads the context and generates a relevant answer
            3. The answer is based ONLY on the provided context

        Why low temperature (0.2)?
            Higher temperature = more creative but less factual.
            For RAG, we want factual answers based on retrieved documents.
            Low temperature makes the model stick to the provided context.
        """
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
