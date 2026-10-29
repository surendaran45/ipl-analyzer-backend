# IPL Analyzer Backend

FastAPI backend for IPL statistics, win-probability predictions, and optional AI-generated cricket analysis.

## Features

- Team, player, venue, head-to-head, and season statistics
- Win-probability predictions using the bundled model and encoders
- Optional AI analysis through the Groq API
- Request validation and consistent HTTP error responses
- Automated API tests with pytest and GitHub Actions

## Requirements

- Python 3.12 recommended
- The model files in `models/`
- Match data in `data/matches_clean.csv`
- A Groq API key only if you want to use `POST /ask-ai`

## Run locally

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Optional; enables the `/ask-ai` endpoint. |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins. Set this to your deployed frontend origin(s) in production. The default `*` is convenient for development but should be restricted in production. |
| `LOG_LEVEL` | Python logging level; defaults to `INFO`. |

Example for a deployed frontend:

```text
CORS_ORIGINS=https://your-frontend.example
GROQ_API_KEY=your-key
```

Configure secrets in your hosting provider's environment settings. Do not commit real API keys.

## Main endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Basic health check |
| GET | `/teams` | List teams |
| GET | `/team/{team_name}` | Team statistics |
| GET | `/h2h?team1=...&team2=...` | Head-to-head statistics |
| GET | `/venues` | List venues |
| GET | `/players` | List players |
| GET | `/player/{player_name}` | Player statistics |
| GET | `/seasons` | Available seasons |
| GET | `/season/{season}` | Season statistics |
| POST | `/win-probability` | Predict the likely winner |
| POST | `/ask-ai` | Request AI-generated analysis |

### Example prediction request

```json
{
  "team1": "Chennai Super Kings",
  "team2": "Mumbai Indians",
  "toss_winner": "Chennai Super Kings",
  "toss_decision": "bat",
  "venue": "M. A. Chidambaram Stadium",
  "season": 2023
}
```

Use values that exist in the bundled training encoders. Unsupported categorical values return HTTP 422 rather than silently mapping to a different category.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The test suite uses fake model/encoder objects for prediction tests, so it does not make external AI calls. GitHub Actions runs the suite on pushes to the default branch and this improvement branch, and on pull requests to the default branch.
