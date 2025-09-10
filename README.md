# Python wrapper for LLM function calling

Function calling provides a powerful and flexible way for Large Language Models (LLMs) such as [OpenAI](https://platform.openai.com/docs/guides/function-calling) and [Anthropic](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) models to interface with external systems (e.g. execute actions on the user's behalf) and access data outside their training data (e.g. search an internal database).

This Python module provides facilities to wrap Python classes and functions such that they can be passed to the `tools` parameter when creating a model with the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools) or [Claude API](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use#specifying-client-tools) in Python. Specifically, it automatically generates a JSON schema from the function signature, doc-string and [Pydantic](https://docs.pydantic.dev/) `Field` definitions of parameters of `BaseModel` type, and unwraps/wraps input/output, filtering and propagating exceptions.

## Features

* generates JSON schema based on function signature
* utilizes `Field` in parameters of types deriving from `BaseModel`
* supports both standard and `async` Python functions
* supports exposing all methods of a Python class eligible for function calling
* marshals input/output between LLM model and Python function
* catches and wraps exceptions

## Example

Below you find a comprehensive example in which we

* define a tool group with invocable functions
* create an asynchronous event stream with OpenAI's Responses API
* process the response stream to collect tool calls
* invoke tools concurrently supplying collected arguments
* transform tool inputs and outputs into messages that we can pass as inputs to subsequent Response API calls

### Define a tool group

First, define a class that derives from `FunctionToolGroup`, and implement member functions that take no parameters, or a single parameter of type `str`, `B` or `list[B]` where `B` derives from `ToolBaseModel`:

<!-- Example 1 -->
```py
class SearchQuery(ToolBaseModel):
    "Finds relevant documents in a database."

    phrase: str = Field(..., description="A search phrase that captures what the user is looking for.")


class SearchResultItem(ToolBaseModel):
    "A document in the database that matches the user's query."

    id: str = Field(..., description="Unique identifier for the document found in the database.")
    content: str = Field(..., description="Document text in Markdown format.")
    similarity: int = Field(..., description="Measures similarity to the user's query on a range from 0 (least similar) to 100 (most similar).")


class SearchToolGroup(FunctionToolGroup):
    "Finds relevant documents in a database."

    connection: object

    def __init__(self, connection: object) -> None:
        self.connection = connection

    async def find_documents(self, query: SearchQuery) -> list[SearchResultItem]:
        "Performs a search on the database to find documents that match the search phrase."

        sql = "SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT ..."
        rows: list[dict[str, typing.Any]] = []
        rows.extend(await self.connection.execute(sql))  # type: ignore
        return [SearchResultItem(id=row["id"], content=row["content"], similarity=row["similarity"]) for row in rows]
```

Next, generate tools by calling `tool_group.async_invocables()` on the tool group, which discovers eligible functions:

<!-- Example 2 -->
```py
# create function tool group
tool_group = SearchToolGroup(connection)

# create invocables
tools = tool_group.async_invocables()
```

### Create a response stream

Pass the list of tools obtained from the tool group to create a response stream using OpenAI's Response API:

<!-- Example 3 -->
```py
async def create_with_tools(
    client: AsyncOpenAI, prompt: str, messages: list[ResponseInputItemParam], tools: list[AsyncInvocable]
) -> AsyncStream[ResponseStreamEvent]:
    """
    Creates a model response stream, enabling a set of tools with function calling.

    :param client: Client proxy for GPT API.
    :param prompt: System prompt for LLM.
    :param messages: Prior messages in the conversation, including user and assistant messages, tool call requests and responses.
    :param tools: Tools to enable with function calling.
    :returns: An asynchronous stream of response events.
    """

    return await client.responses.create(
        stream=True,
        model="gpt-4o-mini",
        instructions=prompt,
        input=messages,
        store=False,
        tools=[
            FunctionToolParam(
                name=tool.name,
                # obtain tool description from function doc-string
                description=tool.description,
                # derive JSON schema from function signature
                parameters=typing.cast(dict[str, object], tool.input_schema()),
                strict=True,
                type="function",
            )
            for tool in tools
        ],
    )
```

### Process response stream

Process the events in the response stream, registering any function calls that the GPT LLM requests to invoke:

<!-- Example 4 -->
```py
async def process_response_stream(events: AsyncStream[ResponseStreamEvent]) -> list[ToolCall]:
    """
    Processes events in a response stream.

    :param events: An asynchronous stream of response events.
    """

    tool_refs: dict[str, ToolRef] = {}
    tool_calls: list[ToolCall] = []

    async for event in events:
        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                # supplies the function name and a unique call identifier
                if event.item.id is not None:
                    tool_refs[event.item.id] = ToolRef(event.item.call_id, event.item.name)
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            # supplies the complete JSON string of function arguments
            tool_ref = tool_refs.pop(event.item_id)
            tool_calls.append(ToolCall(event.item_id, tool_ref.call_id, tool_ref.name, event.arguments))
        else:
            # process other types of events, including message deltas
            pass

    return tool_calls
```

### Execute function call requests

Finally, execute the function call requests by the GPT LLM concurrently and feed back call output as input messages to the GPT LLM:

<!-- Example 5 -->
```py
async def invoke_tools(tools: list[AsyncInvocable], tool_calls: list[ToolCall]) -> list[ResponseInputItemParam]:
    """
    Calls user-defined tools invoked by the LLM.

    :param tools: Tools enabled when creating the response.
    :param tool_calls: Tools invoked by the LLM in the latest response stream.
    :returns: Messages corresponding to tool call requests and responses.
    """

    tool_directory = {tool.name: tool for tool in tools}
    messages: list[ResponseInputItemParam] = []

    # invoke tools concurrently
    tasks: list[Task[str]] = []
    async with TaskGroup() as tg:
        for tool_call in tool_calls:
            tool = tool_directory[tool_call.name]
            tasks.append(tg.create_task(tool(tool_call.args)))

    for tool_call, task in zip(tool_calls, tasks, strict=True):
        call_result = task.result()

        messages.append(
            ResponseFunctionToolCallParam(
                id=tool_call.id,
                call_id=tool_call.call_id,
                name=tool_call.name,
                arguments=tool_call.args,
                type="function_call",
            )
        )
        messages.append(
            FunctionCallOutput(
                id=tool_call.id,
                call_id=tool_call.call_id,
                output=call_result,
                type="function_call_output",
            )
        )

    return messages
```

## Implementation

`ToolBaseModel` configures how the JSON schema is generated by Pydantic such that `additionalProperties` are disallowed to ensure compliance with OpenAI's function tool calling convention in *strict* mode. Otherwise, `ToolBaseModel` is equivalent to a plain Pydantic `BaseModel`.

The implementation makes use of partial type erasure, erasing `B` (a sub-class of `ToolBaseModel`) to `ToolBaseModel`, and `list[B]` to `list[ToolBaseModel]`. This lowers the number of possible input/output combinations that need to be generated, yet allows functions such as `model_json_schema` and `model_validate_json` to be called in context as necessary. This lets us eagerly evaluate some expressions, and elide function calls.
