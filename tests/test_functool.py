import enum
import logging
import unittest
from datetime import date, datetime, time, timedelta
from ipaddress import IPv4Address, IPv6Address
from uuid import UUID

from pydantic import Field

from function_tool import FunctionToolGroup, RuntimeInjected, ToolBaseModel, create_async_invocable, create_invocable, generate_code, get_schema, invocable


class SchemaTypes(ToolBaseModel):
    dt: datetime
    d: date
    t: time
    duration: timedelta
    ipv4: IPv4Address
    ipv6: IPv6Address
    uuid: UUID


@enum.unique
class CardinalDirection(enum.Enum):
    """
    Represents cardinal directions such as north, east, south and west.
    """

    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270


class Wind(ToolBaseModel):
    """
    Characterizes wind speed and direction.
    """

    speed: float = Field(..., description="Wind speed in km/h.")
    direction: CardinalDirection = Field(..., description="Wind direction specified with cardinal directions in degrees.")


class Weather(ToolBaseModel):
    """
    Characterizes weather, i.e. describes the state of the atmosphere at a specific time and place.
    """

    temperature: float = Field(..., description="Temperature in degrees Celsius.")
    wind: Wind | None = Field(..., description="Wind speed and direction.")
    humidity: int = Field(..., description="The amount of moisture in the air expressed as a percentage.")
    precipitation: int = Field(..., description="Probability of precipitation expressed as a percentage.")
    cloudiness: int | None = Field(..., description="The extent to which the atmosphere is covered by cloud expressed as a percentage.")


class Location(ToolBaseModel):
    """
    Represents a location on the globe.
    """

    city: str = Field(..., description="The name of a city that the location represents.")
    country: str = Field(..., description="The name of the country in which the city is located.")


class ForecastRequest(ToolBaseModel):
    """
    Represents a weather forecast request.
    """

    forecast_date: date = Field(..., description="Date that the forecast should apply to.")


class ForecastResponse(ToolBaseModel):
    """
    Represents a weather forecast response.
    """

    weather: Weather


class WeatherForecastTool(FunctionToolGroup):
    def __init__(self) -> None:
        pass

    @classmethod
    def class_function(cls) -> None:
        pass

    @staticmethod
    def static_function() -> None:
        pass

    def _private(self, data: str) -> str:
        "A private function that cannot be invoked externally."

        return data

    @invocable
    def passthru(self, data: str) -> str:
        "A function that consumes and produces unstructured text."

        return data

    @invocable
    def get_location(self) -> Location:
        "A function that produces data that conforms to a schema."

        return Location(city="Budapest", country="Hungary")

    @invocable
    def get_locations(self) -> list[Location]:
        "A function that produces a list of items."

        return [Location(city="Budapest", country="Hungary"), Location(city="Veszprém", country="Hungary")]

    @invocable
    def get_state(self, location: Location) -> Weather:
        "A function that consumes data that conforms to a schema."

        if not location.city:
            raise ValueError("missing city")
        if not location.country:
            raise ValueError("missing country")

        return Weather(temperature=20, wind=Wind(speed=5, direction=CardinalDirection.NORTH), humidity=40, precipitation=30, cloudiness=20)

    @invocable
    def do_action(self, weather: Weather) -> None:
        "A function that succeeds or fails but has no explicit return value."

        pass

    @invocable
    def use_runtime_args(self, weather: Weather, rarg1: RuntimeInjected[int], rarg2: RuntimeInjected[str]) -> str:
        "A function that succeeds or fails but has no explicit return value."
        return f"{weather.humidity}-{rarg1}-{rarg2}"

    @invocable
    def get_forecast(self, request: ForecastRequest) -> ForecastResponse:
        return ForecastResponse(weather=Weather(temperature=30, wind=None, humidity=20, precipitation=0, cloudiness=0))

    def _async_private(self, data: str) -> str:
        "A private function that cannot be invoked externally."

        return data

    @invocable
    async def async_passthru(self, data: str) -> str:
        return data

    @invocable
    async def async_get_location(self) -> Location:
        return self.get_location()

    @invocable
    async def async_get_locations(self) -> list[Location]:
        return self.get_locations()

    @invocable
    async def async_get_state(self, location: Location) -> Weather:
        return self.get_state(location)

    @invocable
    async def async_do_action(self, weather: Weather) -> None:
        pass

    @invocable
    async def async_use_runtime_args(self, weather: Weather, rarg1: RuntimeInjected[int], rarg2: RuntimeInjected[str]) -> str:
        "A function that succeeds or fails but has no explicit return value."
        return f"{weather.humidity}-{rarg1}-{rarg2}"

    @invocable
    async def async_get_forecast(self, request: ForecastRequest) -> ForecastResponse:
        return self.get_forecast(request)


class TestFunctionTool(unittest.TestCase):
    def consume(self, data: SchemaTypes) -> None:
        pass

    def produce(self) -> SchemaTypes:
        return SchemaTypes(
            dt=datetime.now(),
            d=date.today(),
            t=time(23, 59, 59),
            duration=timedelta(microseconds=400),
            ipv4=IPv4Address("127.0.0.1"),
            ipv6=IPv6Address("::1"),
            uuid=UUID("919108f7-52d1-4320-9bac-f847db4148a8"),
        )

    def test_code(self) -> None:
        generate_code()

    def test_invocables(self) -> None:
        tool = WeatherForecastTool()
        self.assertCountEqual(
            tool.invocables(),
            [
                create_invocable(tool.passthru),
                create_invocable(tool.get_location),
                create_invocable(tool.get_locations),
                create_invocable(tool.get_state),
                create_invocable(tool.do_action),
                create_invocable(tool.use_runtime_args),  # type: ignore
                create_invocable(tool.get_forecast),
            ],
        )

    def test_schema(self) -> None:
        self.maxDiff = None
        self.assertEqual(
            get_schema(self.consume),
            {
                "additionalProperties": False,
                "properties": {
                    "dt": {"format": "date-time", "type": "string"},
                    "d": {"format": "date", "type": "string"},
                    "t": {"format": "time", "type": "string"},
                    "duration": {"format": "duration", "type": "string"},
                    "ipv4": {"format": "ipv4", "type": "string"},
                    "ipv6": {"format": "ipv6", "type": "string"},
                    "uuid": {"format": "uuid", "type": "string"},
                },
                "required": ["dt", "d", "t", "duration", "ipv4", "ipv6", "uuid"],
                "type": "object",
            },
        )

        invocable = create_invocable(self.consume)
        self.assertEqual(invocable(self.produce().model_dump_json()), '{"status":"success"}')

    def test_signature(self) -> None:
        self.maxDiff = None
        forecast = WeatherForecastTool()
        self.assertEqual(get_schema(forecast.passthru), {"type": "string"})
        self.assertEqual(get_schema(forecast.get_location), {"type": "object", "properties": {}, "required": [], "additionalProperties": False})
        self.assertEqual(
            get_schema(forecast.get_state),
            {
                "additionalProperties": False,
                "description": "Represents a location on the globe.",
                "properties": {
                    "city": {"description": "The name of a city that the location represents.", "type": "string"},
                    "country": {"description": "The name of the country in which the city is located.", "type": "string"},
                },
                "required": ["city", "country"],
                "type": "object",
            },
        )
        self.assertEqual(
            get_schema(forecast.do_action),
            {
                "$defs": {
                    "CardinalDirection": {
                        "description": "Represents cardinal directions such as north, east, south and west.",
                        "enum": [0, 90, 180, 270],
                        "title": "CardinalDirection",
                        "type": "integer",
                    },
                    "Wind": {
                        "additionalProperties": False,
                        "description": "Characterizes wind speed and direction.",
                        "properties": {
                            "speed": {"description": "Wind speed in km/h.", "type": "number"},
                            "direction": {"$ref": "#/$defs/CardinalDirection", "description": "Wind direction specified with cardinal directions in degrees."},
                        },
                        "required": ["speed", "direction"],
                        "type": "object",
                    },
                },
                "additionalProperties": False,
                "description": "Characterizes weather, i.e. describes the state of the atmosphere at a specific time and place.",
                "properties": {
                    "temperature": {"description": "Temperature in degrees Celsius.", "type": "number"},
                    "wind": {"anyOf": [{"$ref": "#/$defs/Wind"}, {"type": "null"}], "description": "Wind speed and direction."},
                    "humidity": {"description": "The amount of moisture in the air expressed as a percentage.", "type": "integer"},
                    "precipitation": {"description": "Probability of precipitation expressed as a percentage.", "type": "integer"},
                    "cloudiness": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "description": "The extent to which the atmosphere is covered by cloud expressed as a percentage.",
                    },
                },
                "required": ["temperature", "wind", "humidity", "precipitation", "cloudiness"],
                "type": "object",
            },
        )
        self.assertEqual(
            get_schema(forecast.get_forecast),
            {
                "additionalProperties": False,
                "description": "Represents a weather forecast request.",
                "properties": {"forecast_date": {"description": "Date that the forecast should apply to.", "format": "date", "type": "string"}},
                "required": ["forecast_date"],
                "type": "object",
            },
        )

    def test_call(self) -> None:
        forecast = WeatherForecastTool()
        location = forecast.get_location().model_dump_json()
        weather = Weather(temperature=25, wind=None, humidity=0, precipitation=0, cloudiness=0).model_dump_json()
        invocable = create_invocable(forecast.passthru)
        self.assertEqual(invocable.name, "WeatherForecastTool__passthru")
        self.assertEqual(invocable("a message in a bottle"), "a message in a bottle")
        invocable = create_invocable(forecast.get_locations)
        self.assertEqual(invocable(location), '[{"city":"Budapest","country":"Hungary"},{"city":"Veszprém","country":"Hungary"}]')
        invocable = create_invocable(forecast.get_state)
        self.assertEqual(invocable(location), '{"temperature":20.0,"wind":{"speed":5.0,"direction":0},"humidity":40,"precipitation":30,"cloudiness":20}')
        invocable = create_invocable(forecast.get_forecast)
        self.assertEqual(
            invocable('{"forecast_date":"2025-09-09"}'), '{"weather":{"temperature":30.0,"wind":null,"humidity":20,"precipitation":0,"cloudiness":0}}'
        )
        invocable = create_invocable(forecast.use_runtime_args)  # type: ignore
        self.assertEqual(invocable(weather, rarg1=1, rarg2="2"), "0-1-2")
        with self.assertLogs(level=logging.ERROR):
            self.assertEqual(invocable(weather, rarg1=1), '{"status":"failure"}')

    def test_validation(self) -> None:
        forecast = WeatherForecastTool()
        invocable = create_invocable(forecast.get_forecast)
        with self.assertLogs(level=logging.ERROR):
            self.assertEqual(
                invocable('{"forecast_date":"YYYY-MM-DD"}'),
                '{"status":"failure","messages":["Input should be a valid date in the format YYYY-MM-DD, invalid character in year"]}',
            )

    def test_success_failure(self) -> None:
        forecast = WeatherForecastTool()
        weather = Weather(temperature=25, wind=None, humidity=0, precipitation=0, cloudiness=0).model_dump_json()
        invocable = create_invocable(forecast.do_action)
        self.assertEqual(invocable(weather), '{"status":"success"}')
        invocable = create_invocable(forecast.get_state)
        with self.assertLogs(level=logging.ERROR):
            self.assertEqual(invocable('{"city":"","country":""}'), '{"status":"failure"}')

    def test_invocable(self) -> None:
        with self.assertRaises(TypeError):

            @invocable
            def _() -> None:
                pass

        with self.assertRaises(TypeError):

            class A:
                @invocable
                def _private_method(self) -> None:
                    pass

            A()

        with self.assertRaises(TypeError):

            class B:
                @invocable
                @staticmethod
                def static_method() -> None:
                    pass

            B()

        with self.assertRaises(TypeError):

            class C:
                @invocable
                @classmethod
                def class_method(cls) -> None:
                    pass

            C()

        with self.assertRaises(TypeError):

            class D:
                @invocable
                def untyped_method(self, a, b, c):  # type: ignore
                    pass

            D()


class TestAsyncFunctionTool(unittest.IsolatedAsyncioTestCase):
    async def test_invocables(self) -> None:
        tool = WeatherForecastTool()
        self.assertCountEqual(
            tool.async_invocables(),
            [
                create_async_invocable(tool.async_passthru),
                create_async_invocable(tool.async_get_location),
                create_async_invocable(tool.async_get_locations),
                create_async_invocable(tool.async_get_state),
                create_async_invocable(tool.async_do_action),
                create_async_invocable(tool.async_use_runtime_args),  # type: ignore
                create_async_invocable(tool.async_get_forecast),
            ],
        )

    async def test_schema(self) -> None:
        forecast = WeatherForecastTool()
        self.assertEqual(get_schema(forecast.async_passthru), get_schema(forecast.passthru))
        self.assertEqual(get_schema(forecast.async_get_location), get_schema(forecast.get_location))
        self.assertEqual(get_schema(forecast.async_get_state), get_schema(forecast.get_state))
        self.assertEqual(get_schema(forecast.async_do_action), get_schema(forecast.do_action))
        self.assertEqual(get_schema(forecast.async_get_forecast), get_schema(forecast.get_forecast))

    async def test_call(self) -> None:
        forecast = WeatherForecastTool()
        location = forecast.get_location().model_dump_json()
        weather = Weather(temperature=25, wind=None, humidity=0, precipitation=0, cloudiness=0).model_dump_json()
        invocable = create_async_invocable(forecast.async_passthru)
        self.assertEqual(await invocable("a message in a bottle"), "a message in a bottle")
        invocable = create_async_invocable(forecast.async_get_locations)
        self.assertEqual(await invocable(location), '[{"city":"Budapest","country":"Hungary"},{"city":"Veszprém","country":"Hungary"}]')
        invocable = create_async_invocable(forecast.async_get_state)
        self.assertEqual(await invocable(location), '{"temperature":20.0,"wind":{"speed":5.0,"direction":0},"humidity":40,"precipitation":30,"cloudiness":20}')
        invocable = create_async_invocable(forecast.async_do_action)
        self.assertEqual(await invocable(weather), '{"status":"success"}')
        invocable = create_async_invocable(forecast.async_get_state)
        with self.assertLogs(level=logging.ERROR):
            self.assertEqual(await invocable('{"city":"","country":""}'), '{"status":"failure"}')
        invocable = create_async_invocable(forecast.async_use_runtime_args)  # type: ignore
        self.assertEqual(await invocable(weather, rarg1=1, rarg2="2"), "0-1-2")
        with self.assertLogs(level=logging.ERROR):
            self.assertEqual(await invocable(weather, rarg1=1), '{"status":"failure"}')


if __name__ == "__main__":
    unittest.main()
