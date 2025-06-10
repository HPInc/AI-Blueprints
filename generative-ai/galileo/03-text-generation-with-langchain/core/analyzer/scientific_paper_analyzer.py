import logging
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.schema import Document, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, Runnable
from langchain.vectorstores.base import VectorStoreRetriever


class ScientificPaperAnalyzer:
    """
    Class for analyzing scientific papers using LLMs and vector retrievers.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseChatModel,
        prompt_template: Optional[PromptTemplate] = None,
        logging_enabled: bool = False
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()
        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled

        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        else:
            logging.disable(logging.CRITICAL)

        self.chain = self._build_chain()

    def _default_prompt(self) -> PromptTemplate:
        template = (
            "You are analyzing the following scientific paper:\n\n"
            "{context}\n\n"
            "Instruction: {prompt}\n\n"
        )
        return PromptTemplate.from_template(template)

    def _format_docs(self, docs: List[Document]) -> str:
        content = "\n\n".join([d.page_content for d in docs])
        if self.logging_enabled:
            self.logger.info(f"Formatted {len(docs)} documents into context.")
            self.logger.info(f"Context preview: {content[:300]}...")
        return content

    def _query_retriever(self, query: str) -> List[Document]:
        docs = self.retriever.get_relevant_documents(query)
        if self.logging_enabled:
            self.logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
        return docs

    def _build_chain(self) -> Runnable:
        if self.logging_enabled:
            self.logger.info("Building the LangChain chain...")

        return {
            "context": lambda inputs: self._format_docs(self._query_retriever(inputs["prompt"])),
            "prompt": RunnablePassthrough()
        } | self.prompt_template | self.llm | StrOutputParser()

    def analyze(self, prompt: str) -> str:
        if self.logging_enabled:
            self.logger.info(f"Analyzing prompt: '{prompt}'")

        result = self.chain.invoke({"prompt": prompt})

        if self.logging_enabled:
            self.logger.info(f"Raw model output: {str(result)[:300]}...")

        result = result["text"] if isinstance(result, dict) and "text" in result else str(result)
        return result.strip()

    def get_chain(self) -> Runnable:
        return self.chain
