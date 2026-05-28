from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from groq import Groq
from stats_engine import (
    get_team_stats, get_head_to_head, get_venue_stats, get_all_teams,
    get_player_stats, get_all_players, get_player_season_stats,
    get_season_stats, get_all_seasons
)

app = FastAPI(title="IPL Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('models/win_model.pkl')
encoders = joblib.load('models/encoders.pkl')
df = pd.read_csv('data/matches_clean.csv')

code_to_name = {
    'GT': 'Gujarat Titans', 'MI': 'Mumbai Indians',
    'LSG': 'Lucknow Super Giants', 'CSK': 'Chennai Super Kings',
    'RR': 'Rajasthan Royals', 'RCB': 'Royal Challengers Bangalore',
    'DC': 'Delhi Capitals', 'KKR': 'Kolkata Knight Riders',
    'PBKS': 'Punjab Kings', 'SRH': 'Sunrisers Hyderabad',
}
df['winner_full'] = df['winner'].map(code_to_name)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

class WinRequest(BaseModel):
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str
    venue: str
    season: int = 2023

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
    v = sorted(df['venue_name'].dropna().unique().tolist())
    return {"venues": v}

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
        def encode(col, val):
            le = encoders[col]
            if val in le.classes_:
                return le.transform([val])[0]
            return 0
        features = pd.DataFrame([{
            'team1':      encode('team1',    req.team1),
            'team2':      encode('team2',    req.team2),
            'toss_won':   encode('toss_won', req.toss_winner),
            'decision':   encode('decision', req.toss_decision),
            'venue_name': encode('venue_name', req.venue),
            'season':     req.season
        }])
        proba = model.predict_proba(features)[0]
        classes = encoders['winner'].classes_
        result = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
        top2 = sorted(result.items(), key=lambda x: x[1], reverse=True)[:2]
        return {
            "team1": req.team1,
            "team2": req.team2,
            "probabilities": dict(top2),
            "predicted_winner": top2[0][0]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-ai")
async def ask_ai(payload: dict):
    try:
        prompt = payload.get("prompt", "")
        chat = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert IPL cricket analyst. Give sharp, specific, data-driven insights in 4-6 sentences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return {"result": chat.choices[0].message.content}
    except Exception as e:
        return {"error": str(e), "result": "Could not get AI analysis. Check your GROQ_API_KEY."}