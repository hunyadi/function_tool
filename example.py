import asyncio
import os
import typing
from asyncio import Task, TaskGroup
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI, AsyncStream
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_function_call_arguments_done_event import ResponseFunctionCallArgumentsDoneEvent
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call_param import ResponseFunctionToolCallParam
from openai.types.responses.response_input_item_param import ResponseInputItemParam
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.response_output_item_added_event import ResponseOutputItemAddedEvent
from openai.types.responses.response_stream_event import ResponseStreamEvent
from pydantic import Field

from function_tool import AsyncInvocable, FunctionToolGroup, ToolBaseModel


### <<< EXAMPLE 1 <<< ###
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


### >>> EXAMPLE 1 >>> ###


def main() -> None:
    # create GPT client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # create a database connection
    connection = object()

    ### <<< EXAMPLE 2 <<< ###
    # create function tool group
    tool_group = SearchToolGroup(connection)

    # create invocables
    tools = tool_group.async_invocables()
    ### >>> EXAMPLE 2 >>> ###

    asyncio.run(create_stream_execute_tools(client, tools))


async def create_stream_execute_tools(client: AsyncOpenAI, tools: list[AsyncInvocable]) -> None:
    """
    Skeleton for a single iteration of creating a model response stream with active tools, processing the stream, and calling any functions invoked by the LLM.

    This function would typically be called in a loop, feeding the output of function calls back as model input messages.
    """

    # create response stream
    prompt = "..."
    messages: list[ResponseInputItemParam] = []  # prior messages in the conversation
    active_tools = tools  # select tools that are compatible with prompt
    stream = await create_with_tools(client, prompt, messages, active_tools)

    # process response stream
    tool_calls = await process_response_stream(stream)

    # invoke Python implementation and generate tool call request and response messages
    messages = await invoke_tools(tools, tool_calls)


### <<< EXAMPLE 3 <<< ###
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


### >>> EXAMPLE 3 >>> ###


@dataclass
class ToolRef:
    """
    Identifies a function call.

    :param call_id: Couples call request with response.
    :param name: Name of the function to invoke.
    """

    call_id: str
    name: str


@dataclass
class ToolCall:
    """
    Data required to invoke a function with actual arguments.

    :param id: Identifies the function call response item.
    :param call_id: Couples call request with response.
    :param name: Name of the function to invoke.
    :param args: Arguments to pass to the function, serialized as a JSON string.
    """

    id: str
    call_id: str
    name: str
    args: str


### <<< EXAMPLE 4 <<< ###
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


### >>> EXAMPLE 4 >>> ###


### <<< EXAMPLE 5 <<< ###
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


### >>> EXAMPLE 5 >>> ###


if __name__ == "__main__":
    import re
    import textwrap

    examples: dict[int, str] = {}

    with open(Path(__file__).parent / "example.py", "r") as f:
        content = f.read()
        for m in re.finditer(r"^\s*### <<< EXAMPLE (\d+) <<< ###$\n(.+?\n)^\s*### >>> EXAMPLE \1 >>> ###$", content, flags=re.DOTALL | re.MULTILINE):
            examples[int(m.group(1))] = textwrap.dedent(m.group(2))

    def repl(m: re.Match[str]) -> str:
        ref = int(m.group(1))
        return f"<!-- Example {ref} -->\n```py\n{examples[ref]}```"

    with open(Path(__file__).parent / "README.md", "r") as f:
        content = f.read()
        content = re.sub(r"^<!-- Example (\d+) -->$\n^```py$\n.*?^```$", repl, content, flags=re.DOTALL | re.MULTILINE)

    with open(Path(__file__).parent / "README.md", "w") as f:
        f.write(content)
