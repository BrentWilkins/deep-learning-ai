from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, PydanticSingleSelector
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilters


def _get_selector(llm):
    """Use structured tool calls when supported, with a plain-LLM fallback."""
    if llm.metadata.is_function_calling_model:
        return PydanticSingleSelector.from_defaults(llm=llm)
    return LLMSingleSelector.from_defaults(llm=llm)


def _load_nodes(file_path: str):
    """Load a document and fail fast if a PDF was read as raw bytes."""
    documents = SimpleDirectoryReader(
        input_files=[file_path],
        raise_on_error=True,
    ).load_data()
    if not documents:
        raise ValueError(f"No content was extracted from {file_path!r}.")
    if any(
        document.get_content().lstrip().startswith("%PDF-")
        for document in documents
    ):
        raise RuntimeError(
            "The PDF was read as raw bytes. Install `llama-index-readers-file` "
            "and rebuild the documents, nodes, and indexes."
        )

    splitter = SentenceSplitter(chunk_size=1024)
    return splitter.get_nodes_from_documents(documents)


def get_router_query_engine(
    file_path: str,
    llm=None,
    embed_model=None,
    use_async: bool = False,
):
    """Get router query engine."""
    # Reuse the provider-independent models configured by the notebook.
    # Callers can still override either model explicitly.
    if llm is None:
        llm = Settings.llm
    if embed_model is None:
        embed_model = Settings.embed_model

    nodes = _load_nodes(file_path)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=use_async,
        llm=llm,
    )
    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=5,
        llm=llm,
    )

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=("Useful for summarization questions related to MetaGPT"),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=("Useful for retrieving specific context from the MetaGPT paper."),
    )

    query_engine = RouterQueryEngine(
        selector=_get_selector(llm),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        llm=llm,
        verbose=True,
    )
    return query_engine


def get_doc_tools(
    file_path: str,
    name: str,
    llm=None,
    embed_model=None,
    use_async: bool = False,
) -> tuple[FunctionTool, QueryEngineTool]:
    """Get vector query and summary query tools from a document."""
    if llm is None:
        llm = Settings.llm
    if embed_model is None:
        embed_model = Settings.embed_model

    nodes = _load_nodes(file_path)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

    def vector_query(query: str, page_numbers: list[str] | None = None) -> str:
        """Use to answer questions over the MetaGPT paper.

        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """

        page_numbers = page_numbers or []
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

        filters = None
        if metadata_dicts:
            filters = MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR,
            )
        query_engine = vector_index.as_query_engine(
            similarity_top_k=5,
            filters=filters,
            llm=llm,
        )
        response = query_engine.query(query)
        return str(response)

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query,
    )

    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=use_async,
        llm=llm,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            "Use ONLY IF you want to get a holistic summary of MetaGPT. "
            "Do NOT use if you have specific questions over MetaGPT."
        ),
    )

    return vector_query_tool, summary_tool
