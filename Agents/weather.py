import requests
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from dotenv import load_dotenv
import os
from datetime import timedelta, timezone, datetime

load_dotenv()


class WeatherForecast(BaseModel):
    location: str
    description: str
    temperature_celcius: float
    date : str


weather_agent = Agent(
    model="google:gemini-3.5-flash-lite",
    model_settings=ModelSettings(temperature=0),
    output_type=str,
    system_prompt=(
        "You are a helpful weather assistant. Use the 'get_weather_forecast' tool"
        "to find current weather conditions for any city. Provide clean and friendly answers"
        "also greet the user if there is any special occasion, festival, cultural ritual or event on the current day" \
        "eg: - In India it is Dushhera on 20th October 2026, so every Indian city should be greeted with happy dusshera or International days like today is Internation chef day "
    ),
)


#Weather tool with agent
@weather_agent.tool
def get_weather_forecast(ctx: RunContext, city: str) -> WeatherForecast:
    url = "https://api.openweathermap.org/data/2.5/weather"
    api_key = os.getenv("OPENWEATHER_API_KEY")
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    try:
        if response.status_code == 200:
            print(response)
        else:
            print(response.status_code)
        result = response.json()

        observed_at = datetime.fromtimestamp(
        result["dt"],
        tz=timezone(timedelta(seconds=result["timezone"]))
)

        return WeatherForecast(
            location=result["name"],
            description=result["weather"][0]["description"].capitalize(),
            temperature_celcius=result["main"]["temp"],
            date = observed_at.strftime("%d %B %Y")
        )
    except Exception as e:
        raise Exception(e)

question = input("Ask about the weather: ")
agent_response = weather_agent.run_sync(question)
print("\n Forecast:", agent_response.output)
