from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from langchain_agent.context import AgentContext
from langchain_agent.rag.state import RAGState


def ensure_index_node(
    state: RAGState,
    runtime: Runtime[AgentContext],
):
    manager = runtime.context.rag_manager

    if manager.is_indexed:
        return {
            "index_ready": True,
        }

    result = manager.index_repository_knowledge(runtime.context.repository_path)

    if not result["success"]:
        return {
            "index_ready": False,
            "error": result["error"],
        }

    return {
        "index_ready": True,
    }


def route_after_index(
    state: RAGState,
) -> Literal["candidate_retrieve", "__end__"]:
    if state["index_ready"]:
        return "candidate_retrieve"

    return END


def candidate_retrieve_node(
    state: RAGState,
    runtime: Runtime[AgentContext],
):
    rag = runtime.context.rag_manager.repository_rag

    if rag.retrieval_mode == "quality":
        candidate_k = max(
            state["top_k"],
            rag.rerank_candidate_count,
        )
    else:
        candidate_k = state["top_k"]

    results = rag.multi_query_retriever.retrieve(
        query=state["query"],
        top_k=candidate_k,
    )

    return {
        "candidate_results": results,
    }


def rerank_node(
    state: RAGState,
    runtime: Runtime[AgentContext],
):
    rag = runtime.context.rag_manager.repository_rag

    candidates = state["candidate_results"]

    if rag.retrieval_mode == "fast":
        final_results = candidates[: state["top_k"]]

    else:
        assert rag.reranker is not None

        final_results = rag.reranker.rerank(
            query=state["query"],
            results=candidates,
        )[: state["top_k"]]

    return {
        "final_results": final_results,
    }


def build_context_node(
    state: RAGState,
    runtime: Runtime[AgentContext],
):
    rag = runtime.context.rag_manager.repository_rag

    context = rag.context_builder.build(state["final_results"])

    return {
        "query": state["query"],
        "context": context.text,
        "context_character_count": (context.character_count),
    }


def build_rag_graph():
    builder = StateGraph(
        RAGState,
        context_schema=AgentContext,
    )

    builder.add_node(
        "ensure_index",
        ensure_index_node,
    )

    builder.add_node(
        "candidate_retrieve",
        candidate_retrieve_node,
    )

    builder.add_node(
        "rerank",
        rerank_node,
    )

    builder.add_node(
        "build_context",
        build_context_node,
    )

    builder.add_edge(
        START,
        "ensure_index",
    )

    builder.add_conditional_edges(
        "ensure_index",
        route_after_index,
    )

    builder.add_edge(
        "candidate_retrieve",
        "rerank",
    )

    builder.add_edge(
        "rerank",
        "build_context",
    )

    builder.add_edge(
        "build_context",
        END,
    )

    return builder.compile()


rag_graph = build_rag_graph()
