"""
LangChain Utilities

Functions for integrating LangChain with open-source LLMs and RAG systems.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

try:
    from langchain.llms.base import LLM
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.vectorstores import FAISS as LangchainFAISS
    from langchain.vectorstores import Chroma as LangchainChroma
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, some functionality will be limited")


class CustomLLM(LLM):
    """
    Custom LLM class for integrating local models with LangChain.
    """

    model = None
    tokenizer = None
    model_type: str = "phi"

    def __init__(self, model, tokenizer=None, model_type="phi"):
        """
        Initialize the custom LLM.

        Args:
            model: The loaded model
            tokenizer: Optional tokenizer (for transformer models)
            model_type: Model type for prompt formatting
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Please install it with 'pip install langchain'.")

        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type

        # For LangChain type checking
        super().__init__()

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return f"custom_{self.model_type}"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Call the model with the given prompt.

        Args:
            prompt: Prompt text
            stop: Optional list of stop strings

        Returns:
            Generated text
        """
        # Handle different model types
        if self.tokenizer:
            # Transformer model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            # llama.cpp model
            output = self.model(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                stop=stop or []
            )
            response = output["choices"][0]["text"]

        # Check for stop strings
        if stop:
            for stop_str in stop:
                if stop_str in response:
                    response = response[:response.find(stop_str)]

        return response


def create_langchain_documents(texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
    """
    Create LangChain documents from texts and metadata.

    Args:
        texts: List of document texts
        metadatas: List of metadata dictionaries

    Returns:
        List of LangChain Document objects
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed. Please install it with 'pip install langchain'.")

    if not metadatas:
        metadatas = [{} for _ in range(len(texts))]

    return [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]


def create_langchain_retriever(documents: List[Document], embeddings, vector_store_type: str = "faiss"):
    """
    Create a LangChain retriever from documents and embeddings.

    Args:
        documents: List of LangChain Document objects
        embeddings: Embedding model
        vector_store_type: Type of vector store to use

    Returns:
        LangChain retriever
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed. Please install it with 'pip install langchain'.")

    if vector_store_type.lower() == "faiss":
        vector_store = LangchainFAISS.from_documents(documents, embeddings)
    elif vector_store_type.lower() == "chroma":
        vector_store = LangchainChroma.from_documents(documents, embeddings)
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")

    return vector_store.as_retriever(search_kwargs={"k": 5})


def create_qa_chain(llm, retriever, chain_type: str = "stuff"):
    """
    Create a LangChain QA chain.

    Args:
        llm: LangChain LLM
        retriever: LangChain retriever
        chain_type: Type of chain to create

    Returns:
        LangChain QA chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed. Please install it with 'pip install langchain'.")

    prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

    Context:
    {context}

    Question: {question}
    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )


def create_conversational_chain(llm, retriever):
    """
    Create a LangChain conversational chain with memory.

    Args:
        llm: LangChain LLM
        retriever: LangChain retriever

    Returns:
        LangChain conversational chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed. Please install it with 'pip install langchain'.")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
