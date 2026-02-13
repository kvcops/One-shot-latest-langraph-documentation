# LangGraph: Complete Technical Reference

## Overview

LangGraph is a low-level orchestration framework for building stateful, long-running AI agents and workflows. It provides durable execution, human-in-the-loop capabilities, memory management, and streaming support.

**Key Characteristics:**
- Low-level control over agent orchestration
- Built-in persistence layer with checkpointing
- Designed for production deployments
- Supports both graph-based (StateGraph) and functional (Functional API) programming models
- Framework-agnostic (works with or without LangChain)

**Trusted By:** Klarna, Replit, Elastic, and others.

---

## Core Concepts

### Graphs
A **graph** is a directed flow of execution consisting of:
- **Nodes**: Functions that perform work (LLM calls, data processing, tool execution)
- **Edges**: Connections between nodes (normal or conditional)
- **State**: Shared data structure accessible to all nodes
- **START/END**: Special nodes marking entry/exit points

### State
State is a typed dictionary (TypedDict) that:
- Persists across node executions
- Can be updated by nodes
- Supports reducers for merging updates
- Is saved at checkpoints for durability

### Checkpointing & Persistence
- **Checkpoint**: Snapshot of graph state at a super-step
- **Thread**: Unique execution instance identified by `thread_id`
- **Checkpointer**: Storage backend (in-memory, SQLite, Postgres, Redis, MongoDB)
- Enables resumption after interrupts or failures

### Durable Execution
- Automatically saves progress at each super-step
- Allows pausing and resuming workflows
- Supports human-in-the-loop patterns
- Recovers from failures without losing work

---

## Installation

```bash
# Core library
pip install -U langgraph

# With checkpointers
pip install -U langgraph-checkpoint-postgres
pip install -U langgraph-checkpoint-redis
pip install -U langgraph-checkpoint-sqlite

# With stores (long-term memory)
pip install -U langgraph-store-postgres
pip install -U langgraph-store-redis
```

---

## Architecture & Workflow

### Execution Model

1. **Input** → Graph receives initial state
2. **Super-step**: All ready nodes execute in parallel
3. **Checkpoint**: State saved after super-step completes
4. **Routing**: Conditional edges determine next nodes
5. **Repeat** steps 2-4 until END reached

### State Management

**State Schema Definition:**
```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]  # Reducer appends
    count: int  # Normal field (replaces)
```

**State Update Rules:**
- Fields without reducers: replaced
- Fields with reducers: merged (e.g., append, add, custom)
- Updates from multiple nodes in same super-step are combined

---

## API Reference

### StateGraph (Graph API)

**Basic Structure:**
```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    field: str

def node_function(state: State) -> dict:
    return {"field": "updated"}

builder = StateGraph(State)
builder.add_node("node_name", node_function)
builder.add_edge(START, "node_name")
builder.add_edge("node_name", END)
graph = builder.compile()
```

**Key Methods:**
- `add_node(name, function)` - Add node
- `add_edge(from, to)` - Add normal edge
- `add_conditional_edges(from, condition_fn, mapping)` - Add conditional routing
- `compile(checkpointer=None, store=None)` - Build executable graph

**Node Function Signature:**
```python
def node(state: State) -> dict | Command:
    # Return dict for state updates
    # Return Command for routing + updates
    return {"key": "value"}
```

**Command Object:**
```python
from langgraph.types import Command

Command(
    update={"key": "value"},  # State updates
    goto="node_name",  # Next node to execute
    resume="data"  # Resume value for interrupts
)
```

### Functional API

**Basic Structure:**
```python
from langgraph.func import entrypoint, task

@task
def task_function(input):
    return result

@entrypoint()
def workflow(input):
    result1 = task_function(input).result()
    result2 = another_task(result1).result()
    return result2

workflow.invoke(input)
```

**Key Decorators:**
- `@entrypoint()` - Marks main workflow function
- `@task` - Marks cacheable, durable operations

**Task Execution:**
```python
# Synchronous
future = task_function(input)
result = future.result()

# Parallel
futures = [task(x) for x in items]
results = [f.result() for f in futures]
```

### MessagesState

**Predefined State for Chat:**
```python
from langgraph.graph import MessagesState

class State(MessagesState):
    custom_field: str

# MessagesState provides:
# - messages: Annotated[list[AnyMessage], add_messages]
# Automatically handles message deduplication and updates
```

**Message Types:**
```python
from langchain.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage
)
```

### Reducers

**Built-in Reducers:**
```python
import operator

# Append to list
field: Annotated[list, operator.add]

# Sum numbers
field: Annotated[int, operator.add]

# Custom reducer
def custom_reducer(existing, new):
    return merge_logic(existing, new)

field: Annotated[Type, custom_reducer]
```

**Message Reducers:**
```python
from langgraph.graph.message import add_messages

messages: Annotated[list, add_messages]
# Handles message ID deduplication
# Supports RemoveMessage for deletion
```

---

## Checkpointing & Memory

### Short-Term Memory (Thread-Level)

**Setup:**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

**Production Checkpointers:**
```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.sqlite import SqliteSaver

# Postgres
with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

# Redis
with RedisSaver.from_conn_string(redis_uri) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

**Usage:**
```python
config = {"configurable": {"thread_id": "user-123"}}

# First invocation
graph.invoke({"input": "hello"}, config)

# Subsequent invocations use same thread
graph.invoke({"input": "follow-up"}, config)
```

**State Operations:**
```python
# Get current state
state = graph.get_state(config)

# Get state history
history = list(graph.get_state_history(config))

# Update state
graph.update_state(
    config,
    values={"field": "new_value"},
    as_node="node_name"  # Pretend update came from this node
)
```

### Long-Term Memory (Cross-Thread)

**Setup:**
```python
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass

store = InMemoryStore()

@dataclass
class Context:
    user_id: str

graph = builder.compile(store=store)
```

**Access in Nodes:**
```python
from langgraph.runtime import Runtime

async def node(state: State, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    
    # Store data
    await runtime.store.aput(
        namespace, 
        key="memory_id", 
        value={"data": "user info"}
    )
    
    # Retrieve data
    items = await runtime.store.asearch(
        namespace, 
        query="search query",
        limit=5
    )
    
    return {"field": items[0].value}
```

**Production Stores:**
```python
from langgraph.store.postgres import PostgresStore
from langgraph.store.redis import RedisStore

# Similar API to checkpointers
with PostgresStore.from_conn_string(db_uri) as store:
    graph = builder.compile(store=store)
```

**Semantic Search:**
```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["$"]  # Embed all fields
    }
)
```

**Invocation with Context:**
```python
graph.invoke(
    {"input": "data"},
    config={"configurable": {"thread_id": "thread-1"}},
    context=Context(user_id="user-123")
)
```

---

## Execution Methods

### Invoke
```python
result = graph.invoke(
    input={"field": "value"},
    config={"configurable": {"thread_id": "1"}},
    context=Context(...),
    interrupt_before=["node_name"],
    interrupt_after=["node_name"]
)
```

### Stream
```python
for chunk in graph.stream(
    input={"field": "value"},
    config={"configurable": {"thread_id": "1"}},
    stream_mode="updates",  # or "values", "messages", "custom", "debug"
    subgraphs=True
):
    print(chunk)
```

**Stream Modes:**
- `values`: Full state after each step
- `updates`: State changes after each step
- `messages`: LLM tokens + metadata
- `custom`: User-defined streaming data
- `debug`: Maximum execution information

**Multi-Mode Streaming:**
```python
for mode, chunk in graph.stream(input, stream_mode=["updates", "messages"]):
    if mode == "updates":
        # Handle state updates
    elif mode == "messages":
        # Handle LLM tokens
```

---

## Human-in-the-Loop (Interrupts)

### Dynamic Interrupts

**Pause Execution:**
```python
from langgraph.types import interrupt

def approval_node(state: State):
    # Pause and wait for external input
    response = interrupt({
        "question": "Approve this action?",
        "details": state["details"]
    })
    
    if response:
        return {"approved": True}
    return {"approved": False}
```

**Resume Execution:**
```python
from langgraph.types import Command

# Initial run - hits interrupt
result = graph.invoke({"input": "data"}, config)
print(result["__interrupt__"])  # Shows interrupt payload

# Resume with response
graph.invoke(
    Command(resume=True),  # or False, or any JSON-serializable value
    config=config
)
```

### Static Interrupts (Breakpoints)

**At Compile Time:**
```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"]
)
```

**At Runtime:**
```python
graph.invoke(
    input,
    config=config,
    interrupt_before=["node_a"],
    interrupt_after=["node_b"]
)
```

**Resume:**
```python
graph.invoke(None, config=config)  # Resume from last checkpoint
```

### Interrupt Rules

**Critical Rules:**
1. **Never wrap `interrupt()` in bare try/except** - it uses exceptions internally
2. **Keep interrupt order consistent** - calls are matched by index when resuming
3. **Pass only JSON-serializable data** - no functions, classes, or complex objects
4. **Side effects before interrupt must be idempotent** - node re-runs on resume

**Valid Patterns:**
```python
# ✅ Separate interrupt from error-prone code
def node(state):
    response = interrupt("Approve?")
    try:
        risky_operation()
    except SpecificError:
        handle_error()
    return {"field": response}

# ✅ Specific exception handling
def node(state):
    try:
        result = interrupt("Input?")
    except NetworkError:  # Won't catch interrupt exception
        handle_network_issue()
```

**Invalid Patterns:**
```python
# ❌ Bare try/except catches interrupt exception
def node(state):
    try:
        response = interrupt("Approve?")
    except Exception:  # Catches interrupt!
        pass

# ❌ Conditional interrupt order changes
def node(state):
    r1 = interrupt("First")
    if state.get("condition"):
        r2 = interrupt("Second")  # Order inconsistent across runs
    r3 = interrupt("Third")
```

---

## Streaming

### Stream LLM Tokens

**Automatic Token Streaming:**
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1-mini")

def call_model(state):
    response = model.invoke(state["messages"])  # Not .stream()!
    return {"messages": [response]}

# Stream tokens with messages mode
for msg, metadata in graph.stream(
    {"messages": [{"role": "user", "content": "Hi"}]},
    stream_mode="messages",
    config=config
):
    if msg.content:
        print(msg.content, end="")
```

**Filter by Node:**
```python
for msg, metadata in graph.stream(input, stream_mode="messages"):
    if metadata["langgraph_node"] == "specific_node":
        print(msg.content, end="")
```

**Filter by Tag:**
```python
model_1 = init_chat_model("gpt-4.1-mini", tags=["joke"])
model_2 = init_chat_model("gpt-4.1-mini", tags=["poem"])

for msg, metadata in graph.stream(input, stream_mode="messages"):
    if metadata["tags"] == ["joke"]:
        print(msg.content, end="")
```

### Custom Streaming

**Emit Custom Data:**
```python
from langgraph.config import get_stream_writer

def node(state):
    writer = get_stream_writer()
    writer({"custom_key": "progress update"})
    return {"field": "value"}

# Receive custom data
for chunk in graph.stream(input, stream_mode="custom"):
    print(chunk)
```

**Stream from Tools:**
```python
from langchain.tools import tool

@tool
def database_query(query: str):
    writer = get_stream_writer()
    writer({"progress": "Retrieved 0/100 records"})
    # ... perform query
    writer({"progress": "Retrieved 100/100 records"})
    return "results"
```

**Stream Non-LangChain LLMs:**
```python
def call_arbitrary_llm(state):
    writer = get_stream_writer()
    for chunk in custom_streaming_client.stream(state["input"]):
        writer({"llm_chunk": chunk})
    return {"result": "completed"}

for chunk in graph.stream(input, stream_mode="custom"):
    print(chunk["llm_chunk"])
```

### Python < 3.11 Async Limitations

**Manual Config Propagation:**
```python
async def call_model(state, config):  # Accept config
    response = await model.ainvoke(
        state["messages"],
        config  # Pass explicitly
    )
    return {"messages": [response]}
```

**Manual Writer Injection:**
```python
from langgraph.types import StreamWriter

async def node(state, writer: StreamWriter):  # Parameter injection
    writer({"custom": "data"})
    return {"field": "value"}
```

---

## Subgraphs

### Invoke from Node (Different State)

**Pattern: Transform State In/Out**
```python
# Subgraph with different state
class SubgraphState(TypedDict):
    bar: str

subgraph_builder = StateGraph(SubgraphState)
# ... define subgraph
subgraph = subgraph_builder.compile()

# Parent graph
class ParentState(TypedDict):
    foo: str

def call_subgraph(state: ParentState):
    # Transform to subgraph state
    result = subgraph.invoke({"bar": state["foo"]})
    # Transform back to parent state
    return {"foo": result["bar"]}

builder = StateGraph(ParentState)
builder.add_node("call_subgraph", call_subgraph)
```

### Add as Node (Shared State)

**Pattern: Shared State Keys**
```python
# Subgraph shares keys with parent
class State(TypedDict):
    foo: str  # Shared
    bar: str  # Subgraph-only

subgraph_builder = StateGraph(State)
# ... define subgraph
subgraph = subgraph_builder.compile()

# Parent graph
builder = StateGraph(State)
builder.add_node("subgraph", subgraph)  # Add compiled graph directly
builder.add_edge(START, "subgraph")
```

**Checkpointer Propagation:**
```python
# Only compile parent with checkpointer
graph = builder.compile(checkpointer=checkpointer)
# Subgraphs automatically inherit it
```

**Subgraph-Specific Memory:**
```python
subgraph = subgraph_builder.compile(checkpointer=True)
# Subgraph maintains separate checkpointing
```

**Stream Subgraph Outputs:**
```python
for chunk in graph.stream(input, stream_mode="updates", subgraphs=True):
    namespace, data = chunk
    print(f"From {namespace}: {data}")
```

---

## Common Patterns

### Prompt Chaining

**Linear LLM Sequence:**
```python
def generate(state):
    result = llm.invoke(f"Generate content about {state['topic']}")
    return {"content": result.content}

def improve(state):
    result = llm.invoke(f"Improve: {state['content']}")
    return {"content": result.content}

builder.add_edge(START, "generate")
builder.add_edge("generate", "improve")
builder.add_edge("improve", END)
```

### Parallelization

**Concurrent Node Execution:**
```python
def task_a(state): return {"result_a": "data"}
def task_b(state): return {"result_b": "data"}
def task_c(state): return {"result_c": "data"}
def aggregate(state):
    return {"final": combine(state["result_a"], state["result_b"], state["result_c"])}

builder.add_edge(START, "task_a")
builder.add_edge(START, "task_b")
builder.add_edge(START, "task_c")
builder.add_edge("task_a", "aggregate")
builder.add_edge("task_b", "aggregate")
builder.add_edge("task_c", "aggregate")
```

### Routing

**Conditional Branching:**
```python
class Route(BaseModel):
    next_step: Literal["story", "joke", "poem"]

router = llm.with_structured_output(Route)

def route_node(state):
    decision = router.invoke(state["input"])
    return {"decision": decision.next_step}

def route_decision(state):
    return state["decision"]  # Returns node name

builder.add_conditional_edges(
    "route_node",
    route_decision,
    {"story": "story_node", "joke": "joke_node", "poem": "poem_node"}
)
```

### Orchestrator-Worker

**Dynamic Worker Distribution:**
```python
from langgraph.types import Send

class Plan(BaseModel):
    sections: list[str]

def orchestrator(state):
    plan = planner.invoke(state["topic"])
    return {"sections": plan.sections}

def worker(state):
    result = llm.invoke(f"Write about {state['section']}")
    return {"completed": [result.content]}

def assign_workers(state):
    return [Send("worker", {"section": s}) for s in state["sections"]]

builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
builder.add_edge("worker", "synthesizer")
```

### Evaluator-Optimizer

**Iterative Refinement:**
```python
class Feedback(BaseModel):
    grade: Literal["good", "bad"]
    feedback: str

evaluator = llm.with_structured_output(Feedback)

def generator(state):
    result = llm.invoke(state.get("feedback", "Generate output"))
    return {"output": result.content}

def evaluator_node(state):
    feedback = evaluator.invoke(state["output"])
    return {"feedback": feedback.feedback, "grade": feedback.grade}

def route_feedback(state):
    return "generator" if state["grade"] == "bad" else END

builder.add_conditional_edges("evaluator_node", route_feedback)
```

### Tool-Calling Agent

**Standard Agent Loop:**
```python
from langchain.tools import tool

@tool
def search(query: str):
    """Search the web."""
    return perform_search(query)

tools = [search]
model_with_tools = llm.bind_tools(tools)

def call_model(state):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state):
    return "call_tools" if state["messages"][-1].tool_calls else END

builder.add_conditional_edges("call_model", should_continue)
builder.add_edge("call_tools", "call_model")
```

---

## Memory Management

### Trim Messages

**Token-Based Trimming:**
```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def call_model(state):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=1000,
        start_on="human",
        end_on=("human", "tool")
    )
    response = model.invoke(trimmed)
    return {"messages": [response]}
```

### Delete Messages

**Remove Specific Messages:**
```python
from langchain.messages import RemoveMessage

def delete_old_messages(state):
    messages = state["messages"]
    if len(messages) > 10:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:5]]}
    return {}
```

**Remove All Messages:**
```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def clear_history(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

### Summarize Messages

**Rolling Summarization:**
```python
from langmem.short_term import SummarizationNode, RunningSummary

class State(MessagesState):
    context: dict[str, RunningSummary]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=llm,
    max_tokens=1000,
    max_tokens_before_summary=500
)

builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
```

---

## Time Travel

**Replay from Checkpoint:**
```python
# Get state history
states = list(graph.get_state_history(config))

# Select a checkpoint
selected_state = states[2]
checkpoint_id = selected_state.config["configurable"]["checkpoint_id"]

# Resume from that checkpoint
config_with_checkpoint = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": checkpoint_id
    }
}
graph.invoke(None, config=config_with_checkpoint)
```

**Update State at Checkpoint:**
```python
# Modify state at checkpoint
new_config = graph.update_state(
    selected_state.config,
    values={"field": "new_value"},
    as_node="node_name"
)

# Resume from modified checkpoint
graph.invoke(None, config=new_config)
```

---

## Testing

### Test Setup Pattern

```python
import pytest
from langgraph.checkpoint.memory import MemorySaver

@pytest.fixture
def graph():
    builder = StateGraph(State)
    # ... add nodes and edges
    return builder

def test_execution(graph):
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    result = compiled.invoke(
        {"input": "test"},
        config={"configurable": {"thread_id": "test-1"}}
    )
    
    assert result["output"] == "expected"
```

### Test Individual Nodes

```python
def test_node_in_isolation(graph):
    compiled = graph.compile()
    result = compiled.nodes["node_name"].invoke({"input": "test"})
    assert result["output"] == "expected"
```

### Test Partial Execution

```python
def test_partial_path(graph):
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    # Set state as if node1 just completed
    compiled.update_state(
        config={"configurable": {"thread_id": "test"}},
        values={"intermediate": "value"},
        as_node="node1"
    )
    
    # Execute only node2 and node3
    result = compiled.invoke(
        None,
        config={"configurable": {"thread_id": "test"}},
        interrupt_after="node3"
    )
    
    assert result["output"] == "expected"
```

---

## Deployment

### Application Structure

```
my-app/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py          # Graph definition
│   ├── nodes.py          # Node functions
│   ├── tools.py          # Tool definitions
│   └── state.py          # State schema
├── .env                  # Environment variables
├── requirements.txt      # Dependencies
└── langgraph.json       # LangGraph config
```

### Configuration File (langgraph.json)

```json
{
  "dependencies": [
    "langchain_openai",
    "./my_agent"
  ],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": "./.env",
  "store": {
    "index": {
      "embed": "openai:text-embeddings-3-small",
      "dims": 1536,
      "fields": ["$"]
    }
  }
}
```

**Key Fields:**
- `dependencies`: Packages and local modules to install
- `graphs`: Name-to-path mapping of graphs to expose
- `env`: Path to environment variables file
- `store.index`: Semantic search configuration for memory store
- `dockerfile_lines`: Additional Docker commands (optional)

### Database Setup

**Run Migrations:**
```python
# Checkpointer setup
checkpointer.setup()  # or await checkpointer.asetup()

# Store setup
store.setup()  # or await store.asetup()
```

**Connection Pooling:**
```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    # Connection automatically managed
```

---

## Best Practices

### State Design

**Do:**
- Use TypedDict for clear schemas
- Apply reducers for list/accumulator fields
- Keep state keys minimal and focused
- Store raw data, format in nodes

**Don't:**
- Store computed values that can be derived
- Use deeply nested structures
- Put prompt templates in state

### Node Design

**Do:**
- Keep nodes focused on single responsibility
- Return explicit state updates as dicts
- Use Command for routing decisions
- Wrap non-deterministic ops in tasks

**Don't:**
- Mutate state directly
- Perform side effects without idempotency checks
- Mix business logic with routing logic

### Checkpointing

**Do:**
- Use persistent checkpointers in production
- Provide unique thread_ids per conversation
- Leverage checkpoints for debugging
- Configure durability mode appropriately

**Don't:**
- Use InMemorySaver in production
- Share thread_ids across unrelated conversations
- Checkpoint sensitive data without encryption

### Interrupts

**Do:**
- Call interrupt() early in node
- Pass JSON-serializable payloads
- Keep interrupt order consistent
- Make pre-interrupt code idempotent

**Don't:**
- Wrap interrupt() in bare try/except
- Skip interrupts conditionally
- Pass functions or class instances
- Rely on state between interrupt and resume

### Error Handling

**Transient Errors:**
```python
from langgraph.types import RetryPolicy

builder.add_node(
    "api_call",
    api_call_node,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
```

**LLM-Recoverable Errors:**
```python
def tool_node(state):
    try:
        result = execute_tool(state["tool_call"])
    except ToolError as e:
        return {"tool_result": f"Error: {e}"}  # LLM sees error
    return {"tool_result": result}
```

**User-Fixable Errors:**
```python
def node(state):
    if not state.get("required_field"):
        response = interrupt({"error": "Missing required field"})
        return {"required_field": response}
    # Continue processing
```

---

## Common Pitfalls

### Interrupt Anti-Patterns

**❌ Catching Interrupt Exception:**
```python
try:
    response = interrupt("Approve?")
except Exception:  # Catches interrupt signal!
    pass
```

**✅ Specific Exception Handling:**
```python
try:
    response = interrupt("Approve?")
except NetworkError:  # Won't catch interrupt
    handle_network_issue()
```

### State Management Issues

**❌ Forgetting Reducers:**
```python
class State(TypedDict):
    messages: list  # Will be replaced, not appended!
```

**✅ Using Reducers:**
```python
messages: Annotated[list, operator.add]  # Appends
```

### Memory Leaks

**❌ Unbounded State Growth:**
```python
def node(state):
    # Messages accumulate forever
    return {"messages": state["messages"] + [new_message]}
```

**✅ Trim or Summarize:**
```python
def node(state):
    trimmed = trim_messages(state["messages"], max_tokens=1000)
    return {"messages": trimmed + [new_message]}
```

### Checkpointer Misuse

**❌ No Thread ID:**
```python
graph.invoke({"input": "data"})  # State not saved!
```

**✅ Provide Thread ID:**
```python
graph.invoke({"input": "data"}, config={"configurable": {"thread_id": "1"}})
```

---

## Quick Reference

### Essential Imports

```python
# Core
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict, Annotated
import operator

# Checkpointing
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# Memory Store
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

# Messages
from langchain.messages import (
    HumanMessage, AIMessage, SystemMessage,
    ToolMessage, RemoveMessage
)
```

### Basic Graph Template

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    field: str

def node_a(state: State) -> dict:
    return {"field": "updated"}

def node_b(state: State) -> dict:
    return {"field": state["field"] + " again"}

builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())

result = graph.invoke(
    {"field": "initial"},
    config={"configurable": {"thread_id": "1"}}
)
```

### Agent Template

```python
from langgraph.graph import StateGraph, START, MessagesState
from langchain.chat_models import init_chat_model
from langchain.tools import tool

model = init_chat_model("gpt-4.1-mini")

@tool
def search(query: str):
    """Search for information."""
    return f"Results for {query}"

tools = [search]
model_with_tools = model.bind_tools(tools)

def call_model(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: MessagesState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        result = search.invoke(tool_call["args"])
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: MessagesState):
    return "call_tools" if state["messages"][-1].tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("call_tools", call_tools)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue)
builder.add_edge("call_tools", "call_model")

from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())
```

### Execution Patterns

```python
# Basic invoke
result = graph.invoke(input, config={"configurable": {"thread_id": "1"}})

# Stream updates
for chunk in graph.stream(input, config=config, stream_mode="updates"):
    print(chunk)

# Stream LLM tokens
for msg, meta in graph.stream(input, config=config, stream_mode="messages"):
    print(msg.content, end="")

# Multi-mode streaming
for mode, chunk in graph.stream(input, stream_mode=["updates", "messages"]):
    if mode == "updates":
        # Handle state
    elif mode == "messages":
        # Handle tokens

# With subgraphs
for chunk in graph.stream(input, config=config, subgraphs=True):
    namespace, data = chunk
```

---

## Additional Resources

- **LangChain Integration**: `/oss/python/langchain/overview`
- **Multi-Agent Systems**: `/oss/python/langchain/multi-agent`
- **LangSmith Observability**: `/langsmith/home`
- **API Reference**: `https://reference.langchain.com/python/langgraph/`
- **GitHub**: `https://github.com/langchain-ai/langgraph`

---

**End of LangGraph Technical Reference**
