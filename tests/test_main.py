from fastapi.testclient import TestClient

import main


class DummyEncoder:
    def __init__(self, classes):
        self.classes_ = classes

    def transform(self, values):
        return [self.classes_.index(value) for value in values]


class DummyModel:
    def predict_proba(self, features):
        assert list(features.columns) == [
            "team1", "team2", "toss_won", "decision", "venue_name", "season"
        ]
        return [[0.7, 0.3]]


client = TestClient(main.app)


def setup_prediction_fakes(monkeypatch):
    categories = {
        "team1": ["Chennai Super Kings", "Mumbai Indians"],
        "team2": ["Chennai Super Kings", "Mumbai Indians"],
        "toss_won": ["Chennai Super Kings", "Mumbai Indians"],
        "decision": ["bat", "field"],
        "venue_name": ["M. A. Chidambaram Stadium"],
        "winner": ["Chennai Super Kings", "Mumbai Indians"],
    }
    monkeypatch.setattr(
        main, "encoders", {key: DummyEncoder(value) for key, value in categories.items()}
    )
    monkeypatch.setattr(main, "model", DummyModel())


def valid_prediction_payload():
    return {
        "team1": "Chennai Super Kings",
        "team2": "Mumbai Indians",
        "toss_winner": "Chennai Super Kings",
        "toss_decision": "bat",
        "venue": "M. A. Chidambaram Stadium",
        "season": 2023,
    }


def test_root_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "IPL Analyzer API is running!"


def test_win_probability_returns_top_probabilities(monkeypatch):
    setup_prediction_fakes(monkeypatch)

    response = client.post("/win-probability", json=valid_prediction_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_winner"] == "Chennai Super Kings"
    assert body["probabilities"] == {
        "Chennai Super Kings": 70.0,
        "Mumbai Indians": 30.0,
    }


def test_win_probability_rejects_unknown_venue(monkeypatch):
    setup_prediction_fakes(monkeypatch)
    payload = valid_prediction_payload()
    payload["venue"] = "Unknown Stadium"

    response = client.post("/win-probability", json=payload)

    assert response.status_code == 422
    assert "Unsupported value for venue_name" in response.json()["detail"]


def test_win_probability_rejects_same_team_twice():
    payload = valid_prediction_payload()
    payload["team2"] = payload["team1"]

    response = client.post("/win-probability", json=payload)

    assert response.status_code == 422


def test_win_probability_rejects_blank_prompt_and_missing_prompt():
    assert client.post("/ask-ai", json={"prompt": "   "}).status_code == 422
    assert client.post("/ask-ai", json={}).status_code == 422


def test_ask_ai_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(main, "groq_client", None)

    response = client.post("/ask-ai", json={"prompt": "Analyse the last IPL final."})

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
