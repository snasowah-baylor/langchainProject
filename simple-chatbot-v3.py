import os
from math import asin, cos, radians, sin, sqrt

import requests  # type: ignore
from dotenv import load_dotenv  # type: ignore
from langchain.agents import create_agent  # type: ignore
from langchain_core.tools import tool  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


def _geocode(place: str) -> dict | None:
    queries = [place]
    if "," in place:
        # Open-Meteo's geocoder doesn't accept "City, Region" — retry with the head.
        queries.append(place.split(",", 1)[0].strip())
    for q in queries:
        resp = requests.get(GEOCODE_URL, params={"name": q, "count": 1}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if results:
            return results[0]
    return None


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city or place name.

    Returns temperature in Celsius, conditions, and wind speed.
    """
    location = _geocode(city)
    if not location:
        return f"Could not find a location matching '{city}'."

    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
        },
        timeout=10,
    )
    resp.raise_for_status()
    current = resp.json().get("current", {})
    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")
    humidity = current.get("relative_humidity_2m")
    condition = WEATHER_CODES.get(current.get("weather_code"), "unknown conditions")

    name = location.get("name")
    country = location.get("country", "")
    return (
        f"Weather in {name}, {country}: {condition}, {temp}°C, "
        f"humidity {humidity}%, wind {wind} km/h."
    )


@tool
def get_distance(origin: str, destination: str) -> str:
    """Get the great-circle (straight-line) distance between two places in kilometers."""
    a = _geocode(origin)
    b = _geocode(destination)
    if not a:
        return f"Could not find a location matching '{origin}'."
    if not b:
        return f"Could not find a location matching '{destination}'."

    lat1, lon1 = radians(a["latitude"]), radians(a["longitude"])
    lat2, lon2 = radians(b["latitude"]), radians(b["longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    km = 2 * 6371 * asin(sqrt(h))
    miles = km * 0.621371
    return (
        f"Distance from {a['name']}, {a.get('country', '')} to "
        f"{b['name']}, {b.get('country', '')}: {km:.1f} km ({miles:.1f} miles)."
    )


@tool
def lookup_place(query: str) -> str:
    """Look up basic facts about a place: country, region, latitude, longitude, elevation, and population if available."""
    location = _geocode(query)
    if not location:
        return f"Could not find a location matching '{query}'."
    parts = [
        f"{location.get('name')}, {location.get('country', '')}",
        f"region: {location.get('admin1', 'n/a')}",
        f"latitude: {location.get('latitude')}",
        f"longitude: {location.get('longitude')}",
        f"elevation: {location.get('elevation', 'n/a')} m",
    ]
    if location.get("population"):
        parts.append(f"population: {location['population']:,}")
    return " | ".join(parts)


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_agent(
    model=model,
    tools=[get_weather, get_distance, lookup_place],
    system_prompt=(
        "You are a helpful search assistant. You can answer questions about weather, "
        "distances between places, and basic facts about locations. Use the provided "
        "tools when the user asks something that requires real data. Be clear and "
        "concise. If a tool returns an error or no result, say so plainly. If you "
        "don't know an answer and no tool can help, say you don't know."
    ),
)

chat_history: list[dict] = []

print("Search Assistant ready! Ask about weather, distances, or places.")
print("Type 'bye' or 'exit' to end the conversation.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAssistant 🤖: Goodbye! 👋")
        break

    if not user_input:
        continue
    if user_input.lower() in ["bye", "exit"]:
        print("Assistant 🤖: Goodbye! 👋")
        break

    messages = chat_history + [{"role": "user", "content": user_input}]
    result = agent.invoke({"messages": messages})

    reply = result["messages"][-1].content
    print(f"\nAssistant 🤖: {reply}\n")
    print("-" * 60)

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": reply})
