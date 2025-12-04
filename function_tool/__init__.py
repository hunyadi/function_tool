"""
Universal wrapper for Python functions to be used with LLM function calling.

:see: https://github.com/hunyadi/function_tool
"""

__version__ = "0.1.1"
__author__ = "Levente Hunyadi"
__copyright__ = "Copyright 2025, Levente Hunyadi"
__license__ = "MIT"
__maintainer__ = "Levente Hunyadi"
__status__ = "Production"


import enum
import inspect
import json
import logging
import sys
import textwrap
import typing
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from pathlib import Path
from types import CodeType, MethodType
from typing import Annotated, Any, NoReturn, ParamSpec, Self, TypeGuard, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic.config import ConfigDict
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

if sys.version_info >= (3, 12):
    from typing import override as override
else:
    from typing_extensions import override as override

LOGGER = logging.getLogger(__name__)

JsonType = None | bool | int | float | str | dict[str, "JsonType"] | list["JsonType"]


class GenerateJsonSchemaNoTitles(GenerateJsonSchema):
    @override
    def field_title_should_be_set(self, schema) -> bool:  # type: ignore[reportMissingParameterType]
        return False

    @override
    def get_title_from_name(self, name: str) -> str:
        return ""

    def _update_class_schema(self, json_schema: JsonSchemaValue, cls: type[Any], config: ConfigDict) -> None:
        super()._update_class_schema(json_schema, cls, config)
        json_schema.pop("title", None)


class ToolBaseModel(BaseModel):
    "Disables generating `additionalProperties` in JSON schema to ensure compliance with OpenAI's function tool calling convention."

    class Config:
        extra = "forbid"


B = TypeVar("B", bound=ToolBaseModel)
R = TypeVar("R", bound=ToolBaseModel)


class ToolException(Exception):
    "Exceptions that are safe to propagate to the client (i.e. model acting as agent)."

    def __init__(self, message: str) -> None:
        super().__init__(message)

    @property
    def message(self) -> str:
        return typing.cast(str, super().args[0])


def _json_dump(data: JsonType) -> str:
    return json.dumps(data, ensure_ascii=False, check_circular=False, separators=(",", ":"))


@enum.unique
class Status(enum.Enum):
    "Indicates completion status of a wrapped function."

    SUCCESS = "success"
    FAILURE = "failure"


class Response(BaseModel):
    status: Status = Field(..., description="Indicates success or failure for calling a function.")


class _RuntimeInjectedTagType:
    pass


T = typing.TypeVar("T")
_RuntimeInjectedTag = _RuntimeInjectedTagType()
RuntimeInjected = Annotated[T, _RuntimeInjectedTag]


class BaseInvocable(ABC):
    """
    Implements shared behavior for standard (synchronous) and asynchronous wrapped Python functions.
    """

    @property
    def name(self) -> str:
        function_name = self.function.__name__
        if isinstance(self.function, MethodType):
            # use `__`, OpenAI Responses API doesn't allow `.` in function names
            return f"{self.function.__self__.__class__.__name__}__{function_name}"
        else:
            return function_name

    @property
    def description(self) -> str:
        "Produces text for describing the wrapped function when creating an LLM model."

        docstring = self.function.__doc__
        if docstring is None:
            raise TypeError(f"expected: doc-string for {self.name} to add as function tool")

        return textwrap.dedent(docstring)

    @property
    @abstractmethod
    def function(self) -> Callable[..., Any]:
        "Returns the Python function associated with this invocable."

        ...

    @abstractmethod
    def input_schema(self) -> JsonType:
        "Returns the JSON schema for the parameter passed to the function."

        ...

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BaseInvocable) or value.__class__ is not self.__class__:
            return False
        else:
            left = self.function
            right = value.function
            if isinstance(left, MethodType) and isinstance(right, MethodType):
                return left.__func__ is right.__func__ and left.__self__ is right.__self__
            else:
                return left is right

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


def _handle_exception(ex: Exception) -> str:
    "Returns an error message when a validation error is raised."

    response: JsonType
    if isinstance(ex, ToolException):
        response = {"status": Status.FAILURE.value, "message": ex.message}
    elif isinstance(ex, ValidationError):
        response = {
            "status": Status.FAILURE.value,
            "messages": [err["msg"] for err in ex.errors(include_url=False, include_context=False, include_input=False)],
        }
    else:
        response = {"status": Status.FAILURE.value}

    return _json_dump(response)


class Invocable(BaseInvocable):
    """
    Wraps a synchronous function.
    """

    @abstractmethod
    def _execute(self, arg: str, **kwargs: Any) -> str: ...

    def _invoke(self, func: Callable[..., str], arg: str, **kwargs: Any) -> str:
        try:
            return func(arg, **kwargs)
        except Exception as ex:
            LOGGER.exception("function `%s` raised an exception", self.name)
            return _handle_exception(ex)

    def __call__(self, arg: str, **kwargs: Any) -> str:
        """
        Calls an invocable function.

        :param arg: The single input argument to pass to the function, serialized as a JSON string.
        :returns: The result returned by the function, serialized to a JSON string.
        """

        return self._invoke(self._execute, arg, **kwargs)

    def bind(self, **kwargs: Any) -> Self:
        return typing.cast(Self, _PartialInvocable(self, **kwargs))


class AsyncInvocable(BaseInvocable):
    """
    Wraps an asynchronous function.
    """

    async def _invoke(self, func: Callable[..., Awaitable[str]], arg: str, **kwargs: Any) -> str:
        try:
            return await func(arg, **kwargs)
        except Exception as ex:
            LOGGER.exception("function `%s` raised an exception", self.name)
            return _handle_exception(ex)

    @abstractmethod
    async def _execute(self, arg: str, **kwargs: Any) -> str: ...

    async def __call__(self, arg: str, **kwargs: Any) -> str:
        """
        Calls an invocable function asynchronously.

        :param arg: The single input argument to pass to the function, serialized as a JSON string.
        :returns: The result returned by the function, serialized to a JSON string.
        """

        return await self._invoke(self._execute, arg, **kwargs)

    def bind(self, **kwargs: Any) -> Self:
        return typing.cast(Self, _AsyncPartialInvocable(self, **kwargs))


def _check_partial_application(sig: inspect.Signature, **kwargs: Any) -> None:
    """Check that all runtime injected parameters are provided, and no unexpected parameters are given."""
    parameters = (param for param in sig.parameters.values() if _is_runtime_injected_parameter(param))
    for param in parameters:
        if param.name not in kwargs:
            raise TypeError(f"missing runtime injected parameter `{param.name}`")
        kwargs.pop(param.name)
    if kwargs:
        raise TypeError(f"unexpected parameters for partial application: {', '.join(kwargs.keys())}")


class _PartialInvocable:
    """Proxy class for Invocable which bound partial arguments."""

    def __init__(self, obj: Invocable, *args: Any, **kwargs: Any) -> None:
        _check_partial_application(inspect.signature(obj.function), **kwargs)
        self.obj = obj
        self._partial = partial(obj, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._partial(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.obj, name)


class _AsyncPartialInvocable:
    """Proxy class for AsyncInvocable which bound partial arguments."""

    def __init__(self, obj: AsyncInvocable, *args: Any, **kwargs: Any) -> None:
        _check_partial_application(inspect.signature(obj.function), **kwargs)
        self.obj = obj
        self._partial = partial(obj, *args, **kwargs)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self._partial(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.obj, name)


def _object_annotation(object_type: type[Any] | None) -> str:
    if object_type is None:
        return "None"
    elif typing.get_origin(object_type) is list:
        return f"list[{typing.get_args(object_type)[0].__name__}]"
    else:
        return object_type.__name__


def _type_annotation(object_type: type[Any] | None) -> str:
    if object_type is None:
        return "None"
    else:
        return f"type[{_object_annotation(object_type)}]"


def _function_annotation(output_type: type[Any] | None, *, is_async: bool) -> str:
    if output_type is None:
        output_annotation = "None"
    else:
        output_annotation = _object_annotation(output_type)

    if is_async:
        return_annotation = f"Awaitable[{output_annotation}]"
    else:
        return_annotation = output_annotation

    return f"Callable[..., {return_annotation}]"


def is_tool_base_model(arg_type: Any) -> TypeGuard[type[ToolBaseModel]]:
    "True if the argument is a type that derives from `ToolBaseModel`."

    return isinstance(arg_type, type) and issubclass(arg_type, ToolBaseModel)


def is_tool_base_model_list(arg_type: Any) -> TypeGuard[type[Sequence[ToolBaseModel]]]:
    "True if the argument is the type `list[B]` where `B` derives from `ToolBaseModel`."

    if typing.get_origin(arg_type) is not list:
        return False

    (item_type,) = typing.get_args(arg_type)
    return is_tool_base_model(item_type)


FuncToolType = type[ToolBaseModel] | type[Sequence[ToolBaseModel]] | type[str] | None


def _name_for_type(object_type: FuncToolType) -> str:
    if object_type is None:
        return "Void"
    elif object_type is str:
        return "String"
    elif is_tool_base_model(object_type):
        return "Model"
    elif is_tool_base_model_list(object_type):
        return "List"
    else:
        raise TypeError(f"expected: `None`, `B`, `list[B]` or `str` where `B` is a subclass of `{ToolBaseModel.__name__}`; got: {object_type}")


def _name_for_invocable(input_type: FuncToolType, output_type: FuncToolType, *, is_async: bool) -> str:
    "Creates a unique global name for a class deriving from `Invocable` or `AsyncInvocable`."

    input_name = _name_for_type(input_type)
    output_name = _name_for_type(output_type)

    if is_async:
        base = AsyncInvocable.__name__
    else:
        base = Invocable.__name__

    return f"{input_name}Input{output_name}Output{base}"


def _implementation_for_invocable(input_type: FuncToolType, output_type: FuncToolType, *, is_async: bool) -> str:
    "Creates a Python implementation for a class deriving from `Invocable` or `AsyncInvocable`."

    if input_type is None:
        generate_input_schema = '{"type": "object", "properties": {}, "required": [], "additionalProperties": False}'
        cast_input_arg = ""
        pass_parameters = ""
        kwargs_parameters = "**kwargs"
    elif input_type is str:
        generate_input_schema = '{"type": "string"}'
        cast_input_arg = ""
        pass_parameters = "arg"
        kwargs_parameters = ", **kwargs"
    elif is_tool_base_model(input_type):
        generate_input_schema = "self.input_type.model_json_schema(schema_generator=GenerateJsonSchemaNoTitles)"
        cast_input_arg = "input_arg = self.input_type.model_validate_json(arg, strict=True)"
        pass_parameters = "input_arg"
        kwargs_parameters = ", **kwargs"
    elif is_tool_base_model_list(input_type):
        generate_input_schema = f"{TypeAdapter.__name__}(self.input_type).json_schema(schema_generator=GenerateJsonSchemaNoTitles)"
        cast_input_arg = "input_arg = self.input_adapter.validate_json(arg, strict=True)"
        pass_parameters = "input_arg"
        kwargs_parameters = ", **kwargs"
    else:
        raise TypeError(f"expected: zero parameters, a single parameter of type `B`, `list[B]` or `str` where `B` derives from `{ToolBaseModel.__name__}`")

    if output_type is None:
        cast_output_arg = repr(_json_dump({"status": Status.SUCCESS.value}))
    elif output_type is str:
        cast_output_arg = "output_arg"
    elif is_tool_base_model(output_type):
        cast_output_arg = "output_arg.model_dump_json()"
    elif is_tool_base_model_list(output_type):
        cast_output_arg = "self.output_adapter.dump_json(output_arg).decode()"
    else:
        raise TypeError(f"expected: a result that derives from `{ToolBaseModel.__name__}` or `str`")

    if output_type is None:
        assign_result = ""
    else:
        assign_result = "output_arg = "

    function_sig = _function_annotation(output_type, is_async=is_async)
    input_sig = _type_annotation(input_type)
    output_sig = _type_annotation(output_type)

    if is_async:
        async_decl = "async "
        await_decl = "await "
        base = AsyncInvocable.__name__
    else:
        async_decl = ""
        await_decl = ""
        base = Invocable.__name__

    class_code = f"""

class {_name_for_invocable(input_type, output_type, is_async=is_async)}({base}):
    "Implementation for this function has been dynamically generated by :mod:`function_tool`."

    input_type: {input_sig}
    output_type: {output_sig}
    func: {function_sig}
    input_adapter: {TypeAdapter.__name__}[{_object_annotation(input_type)}]
    output_adapter: {TypeAdapter.__name__}[{_object_annotation(output_type)}]

    def __init__(self, func: {function_sig}, input_type: {input_sig}, output_type: {output_sig}) -> None:
        super().__init__()
        self.func = func
        self.input_type = input_type
        self.output_type = output_type
        self.input_adapter = {TypeAdapter.__name__}(self.input_type)
        self.output_adapter = {TypeAdapter.__name__}(self.output_type)

    @property
    @override
    def function(self) -> Callable[..., Any]:
        return self.func

    @override
    def input_schema(self) -> JsonType:
        return {generate_input_schema}

    @override
    {async_decl}def _execute(self, arg: str, **kwargs: Any) -> str:
        {cast_input_arg}
        {assign_result}{await_decl}self.func({pass_parameters}{kwargs_parameters})
        return {cast_output_arg}
"""
    return class_code


def _code_for_invocable(input_type: type | None, output_type: type | None, *, is_async: bool) -> CodeType:
    "Creates a code object for a class deriving from `Invocable` or `AsyncInvocable`."

    code = _implementation_for_invocable(input_type, output_type, is_async=is_async)
    return compile(code, "<string>", "exec")


input_types: list[FuncToolType] = [ToolBaseModel, list[ToolBaseModel], str, None]
output_types: list[FuncToolType] = [ToolBaseModel, list[ToolBaseModel], str, None]


# generate code for the various combinations of input and output parameters and return values
for input_type in input_types:
    for output_type in output_types:
        exec(_code_for_invocable(input_type, output_type, is_async=False))
        exec(_code_for_invocable(input_type, output_type, is_async=True))


def generate_code() -> None:
    """
    Generates code for the various combinations of input and output parameters and return values.

    This utility function lets you inspect the generated code.
    """

    with open(Path(__file__).parent / "code.py", "w") as f:
        f.write(f"""# This file has been generated by a tool.

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import TypeAdapter

from function_tool import {AsyncInvocable.__name__}, {GenerateJsonSchemaNoTitles.__name__}, {Invocable.__name__}, JsonType, {ToolBaseModel.__name__}, override
""")

        count = 0
        for input_type in input_types:
            for output_type in output_types:
                f.write(_implementation_for_invocable(input_type, output_type, is_async=False))
                f.write(_implementation_for_invocable(input_type, output_type, is_async=True))
                count += 2

    print(f"{count} class definitions generated")


def is_func_tool_type(arg_type: type[Any] | None) -> TypeGuard[FuncToolType]:
    return arg_type is None or arg_type is str or is_tool_base_model(arg_type) or is_tool_base_model_list(arg_type)


def _is_runtime_injected_parameter(param: inspect.Parameter) -> bool:
    "True if the parameter is annotated with `RuntimeInjected`."
    if typing.get_origin(param.annotation) is not typing.Annotated:
        return False
    metadata = param.annotation.__metadata__
    return len(metadata) == 1 and metadata[0] is _RuntimeInjectedTag


def _get_parameter_type(sig: inspect.Signature) -> FuncToolType:
    parameters = [param for param in sig.parameters.values() if not _is_runtime_injected_parameter(param)]
    if len(parameters) > 1:
        raise TypeError("expected: zero or one input parameter for a tool function")
    elif len(parameters) > 0:
        input_type = parameters[0].annotation
        if input_type is inspect.Parameter.empty:
            raise TypeError("expected: a typed input parameter for a tool function")
        if not is_func_tool_type(input_type):
            raise TypeError(
                f"expected: a parameter type of `None`, `B`, `list[B]` or `str` where `B` derives from `{ToolBaseModel.__name__}`; got: {input_type}"
            )
        return input_type
    else:
        return None


def _get_return_type(sig: inspect.Signature) -> FuncToolType:
    output_type = sig.return_annotation
    if output_type is inspect.Parameter.empty:
        raise TypeError("expected: a typed result for a tool function")
    elif not is_func_tool_type(output_type):
        raise TypeError(f"expected: a return type of `None`, `B`, `list[B]` or `str` where `B` derives from `{ToolBaseModel.__name__}`; got: {output_type}")
    return output_type


ToolCallable = (
    Callable[[], list[R] | R | str | None]
    | Callable[[B], list[R] | R | str | None]
    | Callable[[list[B]], list[R] | R | str | None]
    | Callable[[str], list[R] | R | str | None]
)


def create_invocable(func: ToolCallable[B, R]) -> Invocable:
    """
    Wraps a standard (synchronous) function into a callable that receives and produces a serialized JSON string.

    :param func: A standard (synchronous function) with a single parameter that derives from `BaseModel`.
    :returns: A standard (synchronous function) that receives and produces a serialized JSON string.
    """

    if inspect.iscoroutinefunction(func):
        raise TypeError(f"use `{create_async_invocable.__name__}` for an asynchronous function")

    sig = inspect.signature(func)
    input_type = _get_parameter_type(sig)
    output_type = _get_return_type(sig)
    return typing.cast(Invocable, globals()[_name_for_invocable(input_type, output_type, is_async=False)](func, input_type, output_type))


AsyncToolCallable = (
    Callable[[], Awaitable[list[R] | R | str | None]]
    | Callable[[B], Awaitable[list[R] | R | str | None]]
    | Callable[[list[B]], Awaitable[list[R] | R | str | None]]
    | Callable[[str], Awaitable[list[R] | R | str | None]]
)


def create_async_invocable(func: AsyncToolCallable[B, R]) -> AsyncInvocable:
    """
    Wraps an asynchronous function into a callable that receives and produces a serialized JSON string.

    :param func: An asynchronous function with a single parameter that derives from `BaseModel`.
    :returns: An synchronous function that receives and produces a serialized JSON string.
    """

    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"use `{create_invocable.__name__}` for a standard (synchronous) function")

    sig = inspect.signature(func)
    input_type = _get_parameter_type(sig)
    output_type = _get_return_type(sig)
    return typing.cast(AsyncInvocable, globals()[_name_for_invocable(input_type, output_type, is_async=True)](func, input_type, output_type))


def get_schema(func: Callable[..., Any]) -> JsonType:
    "Returns a JSON schema for a function signature."

    sig = inspect.signature(func)
    input_type = _get_parameter_type(sig)
    if input_type is None:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    elif is_tool_base_model(input_type):
        return input_type.model_json_schema(schema_generator=GenerateJsonSchemaNoTitles)
    else:
        return typing.cast(JsonType, TypeAdapter(input_type).json_schema(schema_generator=GenerateJsonSchemaNoTitles))


def _is_eligible_function(func: Callable[..., Any]) -> bool:
    "True if the bound or unbound method is eligible for user-defined function calling."

    # skip free functions
    if not isinstance(func, MethodType):
        return False

    # skip magic and private methods
    if func.__name__.startswith("_"):
        return False

    # verify if the method is implemented in a subclass of `FunctionToolGroup` but not `FunctionToolGroup` itself
    if func.__qualname__.startswith(FunctionToolGroup.__name__ + "."):
        return False

    # skip functions with mismatched input signature
    sig = inspect.signature(func)
    parameters = list(sig.parameters.values())

    # remove runtime injected parameters
    parameters = [param for param in parameters if not _is_runtime_injected_parameter(param)]

    if len(parameters) > 1:
        return False
    elif len(parameters) > 0:
        input_param = parameters[0]
        input_type = input_param.annotation
        if input_type is inspect.Parameter.empty or not is_func_tool_type(input_type):
            return False

    # skip functions with mismatched output signature
    output_type = sig.return_annotation
    if output_type is inspect.Parameter.empty or not is_func_tool_type(output_type):
        return False

    return True


def is_eligible_method(func: Callable[..., Any]) -> bool:
    "True if the bound method is eligible for user-defined function calling."

    # include bound methods, skip fields, properties, class and static methods
    if not isinstance(func, MethodType):
        return False

    return _is_eligible_function(func)


class FunctionToolGroup:
    """
    A class whose functions are to be passed to the parameter `tools` when creating a model with the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses).

    Derive from this class when implementing your own class with functions exposed for LLM user-defined function calling.
    """

    def _get_invocable_methods(self, predicate: Callable[[MethodType], bool]) -> list[MethodType]:
        """
        Returns a list of methods filtered by the predicate, and either the presence of an invocable decorator or general eligibility.

        :param predicate: A predicate to filter functions of interest (e.g. asynchronous methods).
        :returns: A list of methods eligible to call as a function tool.
        """

        def _is_candidate_method(method: Callable[..., Any]) -> bool:
            return isinstance(method, MethodType) and method.__self__ is self and predicate(method)

        filtered = [method for _, method in inspect.getmembers(self, predicate=_is_candidate_method)]
        if decorated := [method for method in filtered if getattr(method, "__invocable__", False) is True]:
            # invocable decorator present on at least one method
            return decorated
        else:
            # no invocable decorator on any method, check general eligibility
            return [method for method in filtered if _is_eligible_function(method)]

    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower().removesuffix("tool")

    def invocables(self) -> list[Invocable]:
        "Returns a list of standard (synchronous) functions this tool exposes."

        methods = self._get_invocable_methods(lambda func: not inspect.iscoroutinefunction(func))
        if not methods:
            raise ValueError("expected: at least one exposed synchronous function; got: zero")
        return [create_invocable(method) for method in methods]

    def async_invocables(self) -> list[AsyncInvocable]:
        "Returns a list of asynchronous functions this tool exposes."

        methods = self._get_invocable_methods(lambda func: inspect.iscoroutinefunction(func))
        if not methods:
            raise ValueError("expected: at least one exposed asynchronous function; got: zero")
        return [create_async_invocable(method) for method in methods]


Params = ParamSpec("Params")
Return = TypeVar("Return")


def invocable(func: Callable[Params, Return]) -> Callable[Params, Return]:
    """
    Decorator to verify if a member function can be passed to `create_invocable` or `create_async_invocable`.

    If the decorator is added to a function not eligible for function calling, it will trigger a :exc:`TypeError`.

    Functions eligible for function calling must be:

    * public
    * instance methods
    * have zero or a single input parameter of `B`, `list[B]` or `str` where `B` is a subclass of :class:`ToolBaseModel`
    * have a return type `None`, `B`, `list[B]` or `str` where `B` is a subclass of :class:`ToolBaseModel`

    Functions eligible for function calling **cannot** be:

    * magic (e.g. `__str__` or `__repr__`)
    * private
    * class methods
    * static methods
    """

    def raise_exception(kind: str) -> NoReturn:
        raise TypeError(f"expected: a method eligible for function calling; got {kind}: {func.__name__}")

    if isinstance(func, staticmethod):
        raise_exception("static method")

    if isinstance(func, classmethod):
        raise_exception("class method")

    # skip magic methods
    if func.__name__.startswith("__") and func.__name__.endswith("__"):
        raise_exception("magic method")

    # skip private methods
    if func.__name__.startswith("_"):
        raise_exception("private method")

    # verify if the method is implemented in a subclass of `FunctionToolGroup` but not `FunctionToolGroup` itself
    if func.__qualname__.startswith(FunctionToolGroup.__name__ + "."):
        raise_exception(f"method implemented directly in {FunctionToolGroup.__name__}")

    # fetch method signature
    sig = inspect.signature(func)
    parameters = list(sig.parameters.values())

    # remove runtime injected parameters
    parameters = [param for param in parameters if not _is_runtime_injected_parameter(param)]

    # method object binding has not yet taken place when decorator function is called
    if not isinstance(func, MethodType):
        if not parameters or parameters[0].name != "self":
            # unbound methods must have at least the parameter `self`
            raise_exception("function whose first parameter is not `self`")
        parameters = parameters[1:]

    # skip functions with mismatched input signature
    if len(parameters) > 1:
        raise_exception("function with more than one parameter")
    elif len(parameters) > 0:
        input_param = parameters[0]
        input_type = input_param.annotation
        if input_type is inspect.Parameter.empty:
            raise_exception(f"untyped parameter `{input_param.name}` in function")
        if not is_func_tool_type(input_type):
            raise_exception(f"parameter `{input_param.name}` of wrong type in function")

    # skip functions with mismatched output signature
    output_type = sig.return_annotation
    if output_type is inspect.Parameter.empty:
        raise_exception("function with untyped return value")
    if not is_func_tool_type(output_type):
        raise_exception("function with wrong return type")

    # enable runtime introspection
    setattr(func, "__invocable__", True)  # noqa: B010
    return func
