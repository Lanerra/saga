"""Offline lifecycle/API contract probes using synthetic state and narrow fakes."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from core.langgraph.content_manager import ContentManager, save_chapter_plan, save_draft
import core.langgraph.nodes.revision_node as revision
import core.langgraph.nodes.finalize_node as finalize
import core.langgraph.nodes.context_scene_retrieval as context
from core.langgraph.subgraphs import validation

results = {}

class ToyState(TypedDict):
    first_runs: int
    second_runs: int


def first(state):
    return {'first_runs': state['first_runs'] + 1}


def second(state):
    return {'second_runs': state['second_runs'] + 1}


def make_graph():
    graph = StateGraph(ToyState)
    graph.add_node('first', first)
    graph.add_node('second', second)
    graph.add_edge(START, 'first')
    graph.add_edge('first', 'second')
    graph.add_edge('second', END)
    return graph.compile(checkpointer=InMemorySaver(), interrupt_before=['second'])


for mode in ['state_as_new_input', 'none_as_resume']:
    graph = make_graph()
    config = {'configurable': {'thread_id': mode}}
    graph.invoke({'first_runs': 0, 'second_runs': 0}, config)
    snapshot = graph.get_state(config)
    graph.invoke(dict(snapshot.values) if mode == 'state_as_new_input' else None, config)
    final = graph.get_state(config)
    results[mode] = {'values': final.values, 'next': list(final.next)}


async def main():
    with tempfile.TemporaryDirectory(prefix='saga-lifecycle-probe-') as temporary:
        manager = ContentManager(temporary)
        plan = save_chapter_plan(manager, [{'title':'Synthetic scene','pov_character':'A','setting':'Lab','plot_point':'Discovery','conflict':'Doubt','outcome':'Decision'}], 1)
        state = {'project_id':'synthetic','project_dir':temporary,'current_chapter':1,'chapter_plan_ref':plan,'iteration_count':0,'max_iterations':2,'contradictions':[]}
        calls = []
        async def fail_rollback(*args, **kwargs):
            calls.append('rollback_failed')
            raise RuntimeError('synthetic rollback failure')
        async def generated(*args, **kwargs):
            calls.append('guidance_generated')
            return 'Synthetic revision guidance only.', {}
        revision.neo4j_manager = SimpleNamespace(execute_cypher_batch=fail_rollback)
        revision.llm_service = SimpleNamespace(count_tokens=lambda *args: 10, async_call_llm=generated)
        revised = await revision.revise_chapter(state)
        results['revision_after_failed_rollback'] = {'calls':calls,'fatal':revised.get('has_fatal_error',False),'guidance_written':bool(revised.get('revision_guidance_ref'))}
        async def fail_file(*args, **kwargs):
            raise OSError('synthetic filesystem failure')
        async def no_embedding(*args, **kwargs):
            return None
        database_calls=[]
        async def save_db(**kwargs):
            database_calls.append(kwargs)
        finalize._save_chapter_to_filesystem = fail_file
        finalize.llm_service = SimpleNamespace(async_get_embedding=no_embedding)
        finalize.save_chapter_data_to_db = save_db
        state['draft_ref'] = save_draft(manager,'Synthetic chapter prose.',1)
        final = await finalize.finalize_chapter(state)
        results['finalize_after_failed_file_write'] = {'fatal':final.get('has_fatal_error',False),'database_calls':len(database_calls),'canonical_manuscript_exists':(Path(temporary)/'chapters/chapter_001.md').exists(),'db_receives_full_prose':any('text' in k or 'draft' in k for row in database_calls for k in row)}
        context.PREVIOUS_SCENES_TOKEN_BUDGET=10
        context.SUMMARY_MAX_TOKENS=10
        context.count_tokens=lambda text, model: len(text.split())
        drafts=['one two three four five six seven eight']*3
        rendered=await context.get_previous_scenes_context(state,drafts,[{'title':str(i)} for i in range(3)],3,'fake','fake',manager)
        results['previous_scene_budget']={'declared_total_tokens':10,'included_prose_tokens':sum(len(d.split()) for d in drafts),'returned_context_tokens':len(rendered.split())}
        existing=[{'source_name':'A','target_name':'B','rel_type':'LOVES','chapter':2},{'source_name':'A','target_name':'B','rel_type':'HATES','chapter':1}]
        async def unordered(*args,**kwargs):return existing
        validation.neo4j_manager=SimpleNamespace(execute_read_query=unordered)
        loaded=await validation._fetch_validation_data(2)
        issues=await validation._check_relationship_evolution([{'source_name':'A','target_name':'B','relationship_type':'LOVES'}],2,loaded['relationships'])
        results['validation_current_row_first']={'selected':loaded['relationships'][('A','B')],'issues':len(issues)}
        existing.reverse()
        loaded=await validation._fetch_validation_data(2)
        issues=await validation._check_relationship_evolution([{'source_name':'A','target_name':'B','relationship_type':'LOVES'}],2,loaded['relationships'])
        results['validation_historical_row_first']={'selected':loaded['relationships'][('A','B')],'issues':len(issues)}


asyncio.run(main())
print('PROBE_RESULTS_JSON='+json.dumps(results,sort_keys=True))
