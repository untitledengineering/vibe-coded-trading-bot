"""News sentiment scoring for Indian equity headlines.

Two-implementation interface so plumbing works regardless of whether you've
set up an Anthropic API key yet:

    - ClaudeSentimentScorer:  real scoring via Claude Haiku with prompt caching.
    - StubSentimentScorer:    returns neutral (0, 0, 0). Tests and bootstrapping.

Selection is by env var: if ANTHROPIC_API_KEY is present, Claude is used;
otherwise the stub. Same `SentimentResult` shape returned either way so
downstream code is identical.

The scored output is persisted on the existing `news` row (no separate table —
keeps reads single-query and lets us re-score by clearing sentiment_at).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol

import aiosqlite

from src.utils.config import DB_PATH
from src.utils.logger import logger


CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 200

# The rubric is hoisted out of the function so it can be cached at the API level
# (every Claude call sends the same system prompt → prompt-cache hit after #1).
SYSTEM_PROMPT = """You score Indian equity news headlines for short-horizon (15-minute) market impact.

Output ONLY valid JSON with these keys:
  {"score": <float -1..1>, "confidence": <float 0..1>, "decay_minutes": <int>}

score (directional impact for the named ticker, or for the broad Indian equity market if no ticker):
  +1.0  Strong bullish catalyst — earnings beat, major contract win, M&A target, positive regulatory ruling
  +0.5  Mildly positive — analyst upgrade, sector rally, peer earnings beat
   0.0  Neutral or generic market commentary
  -0.5  Mildly negative — analyst downgrade, sector weakness, peer earnings miss
  -1.0  Strong bearish catalyst — fraud allegation, regulator action, earnings miss, accounting concern

confidence:
   1.0  Unambiguous, material, directly about the named ticker (or clearly market-wide)
   0.5  Possibly relevant, indirect, or tangential
   0.0  Not actually about this ticker, pure clickbait, or unparseable

decay_minutes (how long the directional signal stays useful for intraday trading):
   10   Pure technical / "stock rose X%" / already-priced-in headlines
   30   Standard intraday news, analyst notes, sector moves
  180   Earnings releases, M&A announcements, regulator action

Output JSON only. No prose, no markdown, no code fences."""


@dataclass
class SentimentResult:
    score: float
    confidence: float
    decay_minutes: int
    model: str  # which scorer produced this; useful for re-scoring decisions


class SentimentScorer(Protocol):
    async def score(self, headline: str, ticker: Optional[str] = None) -> SentimentResult: ...


# ---------- Stub ----------

class StubSentimentScorer:
    """Neutral scorer. Used when no API key is configured and in tests that
    don't want to mock Anthropic. Marks results with model='stub' so downstream
    code can opt to ignore them."""

    model_name = "stub"

    async def score(self, headline: str, ticker: Optional[str] = None) -> SentimentResult:
        del headline, ticker  # interface params; deliberately ignored by the stub
        return SentimentResult(score=0.0, confidence=0.0, decay_minutes=0, model=self.model_name)


# ---------- Claude ----------

class ClaudeSentimentScorer:
    """Async Anthropic client with prompt caching on the system message."""

    def __init__(self, api_key: str, model: str = CLAUDE_MODEL):
        # Local import so `import sentiment` doesn't fail when anthropic is uninstalled.
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    async def score(self, headline: str, ticker: Optional[str] = None) -> SentimentResult:
        user = f"Headline: {headline}\nTicker: {ticker or 'general market'}"
        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            logger.error(f"Claude sentiment call failed: {type(e).__name__}")
            return SentimentResult(0.0, 0.0, 0, f"{self.model}:error")

        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()
        return _parse_sentiment_json(text, self.model)


def _parse_sentiment_json(text: str, model: str) -> SentimentResult:
    """Tolerant parser. LLMs occasionally wrap JSON in code fences despite instructions."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a ```json ... ``` fence if present.
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        cleaned = cleaned.rstrip("`").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Sentiment parse failed; falling back to neutral. Raw: {text[:120]!r}")
        return SentimentResult(0.0, 0.0, 0, f"{model}:unparseable")

    try:
        score = float(data["score"])
        confidence = float(data["confidence"])
        decay = int(data["decay_minutes"])
    except (KeyError, TypeError, ValueError):
        logger.warning(f"Sentiment shape invalid; falling back to neutral. Raw: {text[:120]!r}")
        return SentimentResult(0.0, 0.0, 0, f"{model}:invalid")

    # Clamp to valid ranges so a misbehaving model can't poison features downstream.
    score = max(-1.0, min(1.0, score))
    confidence = max(0.0, min(1.0, confidence))
    decay = max(0, min(720, decay))
    return SentimentResult(score=score, confidence=confidence, decay_minutes=decay, model=model)


def get_default_scorer() -> SentimentScorer:
    """Pick the real scorer if a key is present, else the stub. Single place to
    flip the source so the rest of the code never reads ANTHROPIC_API_KEY directly."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        logger.info("ANTHROPIC_API_KEY not set; using StubSentimentScorer")
        return StubSentimentScorer()
    return ClaudeSentimentScorer(api_key=key)


# ---------- Batch driver ----------

async def _fetch_unscored(db, limit: int) -> List[tuple]:
    async with db.execute(
        """
        SELECT id, headline, instrument_key
        FROM news
        WHERE sentiment_at IS NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ) as cur:
        return await cur.fetchall()


def _ticker_for(instrument_key: Optional[str]) -> Optional[str]:
    """Best-effort trading-symbol extraction from an instrument_key like
    'NSE_EQ|INE002A01018'. Returns the ISIN suffix when present, which Claude
    accepts as a disambiguator. Returns None for market-wide headlines."""
    if not instrument_key:
        return None
    return instrument_key.split("|", 1)[-1] if "|" in instrument_key else instrument_key


async def score_unscored_news(
    scorer: SentimentScorer,
    limit: int = 100,
    concurrency: int = 5,
    db_path: Optional[str] = None,
) -> int:
    """Find up to `limit` unscored headlines and score them in parallel batches.
    Returns the count of newly scored rows."""
    db_path = db_path or DB_PATH
    sem = asyncio.Semaphore(concurrency)
    scored = 0

    async def _score_one(news_id: int, headline: str, instrument_key: Optional[str]):
        nonlocal scored
        async with sem:
            ticker = _ticker_for(instrument_key)
            result = await scorer.score(headline, ticker=ticker)
            # Hold the semaphore slot for 2s after each call so we stay well under
            # Tier-1's 50 RPM cap (2 concurrent × 1 call/2s = ~30 RPM max).
            await asyncio.sleep(2.0)
            now = int(time.time())
            async with aiosqlite.connect(db_path) as udb:
                await udb.execute(
                    """
                    UPDATE news
                    SET sentiment_score = ?,
                        sentiment_confidence = ?,
                        sentiment_decay_minutes = ?,
                        sentiment_model = ?,
                        sentiment_at = ?
                    WHERE id = ?
                    """,
                    (result.score, result.confidence, result.decay_minutes,
                     result.model, now, news_id),
                )
                await udb.commit()
            scored += 1

    async with aiosqlite.connect(db_path) as db:
        rows = await _fetch_unscored(db, limit)

    if not rows:
        return 0

    await asyncio.gather(*(_score_one(*row) for row in rows))
    return scored


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Score unscored news headlines.")
    parser.add_argument("--limit", type=int, default=20, help="Max headlines to score this run")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Parallel in-flight Claude calls. Keep modest to stay under rate limit.",
    )
    args = parser.parse_args()

    scorer = get_default_scorer()
    if isinstance(scorer, StubSentimentScorer):
        logger.warning(
            "Running with StubSentimentScorer (all scores will be 0.0). "
            "Set ANTHROPIC_API_KEY for real scoring."
        )
    n = asyncio.run(
        score_unscored_news(scorer, limit=args.limit, concurrency=args.concurrency)
    )
    print(f"Scored {n} headlines with {scorer.model_name}.")


if __name__ == "__main__":
    _cli()
