"""LangGraph fan-out over isolated page processes; stage logic stays in workers."""
from __future__ import annotations

from itertools import islice
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class PageState(TypedDict, total=False):
    tasks: list
    task: Any
    completed: Annotated[list, operator.add]


def page_results(pool, worker, tasks, concurrency, cancel):
    """Stream completed pages with at most one bounded window in memory.

    LangGraph threads only wait on futures. PyMuPDF, SQLite handles and model
    clients stay in spawned processes with the existing shared HTTP limiter.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError('LangGraph concurrency must be between 1 and 32')

    def dispatch(state):
        return [Send('process_page', {'task': task}) for task in state['tasks']]

    def process_page(state):
        task = state['task']
        return {'completed': [(task, pool.submit(worker, task).result())]}

    builder = StateGraph(PageState)
    builder.add_node('process_page', process_page)
    builder.add_conditional_edges(START, dispatch, ['process_page'])
    builder.add_edge('process_page', END)
    graph = builder.compile()
    pending = iter(tasks)
    try:
        while batch := list(islice(pending, concurrency)):
            for update in graph.stream({'tasks': batch, 'completed': []},
                                       config={'max_concurrency': concurrency}, stream_mode='updates'):
                for result in update.get('process_page', {}).get('completed', []):
                    yield result
    except BaseException:
        cancel()
        raise
