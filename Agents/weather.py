import requests
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from dotenv import load_dotenv
import os

load_dotenv()


class WeatherForecast(BaseModel):
    location: str
    description: str
    temperature_celcius: float


weather_agent = Agent(
    model="google:gemini-3.5-flash-lite",
    model_settings=ModelSettings(temperature=0),
    output_type=str,
    system_prompt=(
        "You are a helpful weather assistant. Use the 'get_weather_forecast' tool"
        "to find current weather conditions for any city. Provide clean and friendly answers"
    ),
)


#Weather tool with agent
@weather_agent.tool
def get_weather_forecast(ctx: RunContext, city: str) -> WeatherForecast:
    url = "https://api.openweathermap.org/data/2.5/weather"
    api_key = os.getenv("OPENWEATHER_API_KEY")
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    print(response)
    # result = response.json()

    # return WeatherForecast(
    #     location=result["name"],
    #     description=result["weather"][0]["description"].capitalize(),
    #     temperature_celcius=result["main"]["temp"],
    # )

# question = input("Ask about the weather: ")
# agent_response = weather_agent.run_sync(question)
# print("\n Forecast:", agent_response.output)
