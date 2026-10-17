# AI Model Router Backend

FastAPI gateway that routes requests across five real free-tier models. There is intentionally

## Models
1. Google Gemini 2.5 Flash via Google AI API
2. NVIDIA Nemotron 3 Ultra (free) via OpenRouter
3. Google Gemma 4 31B (free) via OpenRouter
4. Google Gemma 4 26B A4B (free) via OpenRouter
5. Qwen 3.8 27B (free) via OpenRouter

Free-tier availability/rate limits can change. The model catalog is kept in `app/models.py` so it can be updated without changing the router.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```
Put your keys in `.env`:
```env
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
```
Then:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs
