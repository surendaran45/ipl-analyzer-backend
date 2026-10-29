import logging
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stats_engine import (
    get_team_stats,
    get_head_to_head,
    get_venue_stats,
    get_all_teams,
    get_player_stats,
    get_all_players,
    get_player_season_stats,
    get_season_stats,
    get_all_seasons,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

app = FastAPI(title="IPL Analyzer API", version="1.1.0")

# Set CORS_ORIGINS to a comma-separated list of trusted frontend origins in production.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

model = joblib.load("models/win_model.pkl")
encoders = joblib.load("models/encoders.pkl")
df = pd.read_csv("data/matches_clean.csv")

code_to_name = {
    "GT": "Gujarat Titans",
    "MI": "Mumbai Indians",
    "LSG": "Lucknow Super Giants",
    "CSK": "Chennai Super Kings",
    "RR": "Rajasthan Royals",
    "RCB": "Royal Challengers Bangalore",
    "DC": "Delhi Capitals",
    "KKR": "Kolkata Knight Riders",
    "PBKS": "Punjab Kings",
    "SRH": "Sunrisers Hyderabad",
}
df["winner_full"] = df["winner"].map(code_to_name)

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


class WinRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    team1: str = Field(min_length=1, max_length=100)
    team2: str = Field(min_length=1, max_length=100)
    toss_winner: str = Field(min_length=1, max_length=100)
    toss_decision: str = Field(min_length=1, max_length=30)
    venue: str = Field(min_length=1, max_length=200)
    season: int = Field(default=2023, ge=2008, le=2100)

    @model_validator(mode="after")
    def teams_must_be_different(self):
        if self.team1.casefold() == self.team2.casefold():
            raise ValueError("team1 and team2 must be different teams")
        return self


class AskAIRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=4000)


def encode_known_category(column: str, value: str) -> int:
    """Encode a category only when it was present during model training."""
    encoder = encoders[column]
    if value not in encoder.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported value for {column}: {value}",
        )
    return int(encoder.transform([value])[0])


@app.get("/")
def root():
    return {"message": "IPL Analyzer API is running!"}


@app.get("/teams")
def teams():
    return {"teams": get_all_teams()}


@app.get("/team/{team_name}")
def team_stats(team_name: str):
    return get_team_stats(team_name)


@app.get("/h2h")
def head_to_head(team1: str, team2: str):
    return get_head_to_head(team1, team2)


@app.get("/venue")
def venue_stats(name: str):
    return get_venue_stats(name)


@app.get("/venues")
def venues():
    return {"venues": sorted(df["venue_name"].dropna().unique().tolist())}


@app.get("/players")
def players():
    return {"players": get_all_players()}


@app.get("/player/{player_name}")
def player_stats(player_name: str):
    return get_player_stats(player_name)


@app.get("/player/{player_name}/seasons")
def player_seasons(player_name: str):
    return {"seasons": get_player_season_stats(player_name)}


@app.get("/seasons")
def seasons():
    return {"seasons": get_all_seasons()}


@app.get("/season/{season}")
def season_stats(season: int):
    return get_season_stats(season)


@app.post("/win-probability")
def win_probability(req: WinRequest):
    try:
        features = pd.DataFrame(
            [
                {
                    "team1": encode_known_category("team1", req.team1),
                    "team2": encode_known_category("team2", req.team2),
                    "toss_won": encode_known_category("toss_won", req.toss_winner),
                    "decision": encode_known_category("decision", req.toss_decision),
                    "venue_name": encode_known_category("venue_name", req.venue),
                    "season": req.season,
                }
            ]
        )
        probabilities = model.predict_proba(features)[0]
        classes = encoders["winner"].classes_
        result = {
            str(label): round(float(probability) * 100, 1)
            for label, probability in zip(classes, probabilities)
        }
        top_two = sorted(result.items(), key=lambda item: item[1], reverse=True)[:2]
        if not top_two:
            raise RuntimeError("The prediction model returned no classes")
        return {
            "team1": req.team1,
            "team2": req.team2,
            "probabilities": dict(top_two),
            "predicted_winner": top_two[0][0],
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Win-probability prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Could not calculate win probability. Please try again later.",
        ) from None


@app.post("/ask-ai")
async def ask_ai(payload: AskAIRequest):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="AI analysis is unavailable because GROQ_API_KEY is not configured.",
        )

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert IPL cricket analyst. Give sharp, "
                        "specific, data-driven insights in 4-6 sentences."
                    ),
                },
                {"role": "user", "content": payload.prompt},
            ],
            model="llama-3.3-70b-versatile",
        )
        return {"result": chat.choices[0].message.content}
    except Exception:
        logger.exception("Groq AI analysis request failed")
        raise HTTPException(
            status_code=502,
            detail="Could not get AI analysis from the provider. Please try again later.",
        ) from None
