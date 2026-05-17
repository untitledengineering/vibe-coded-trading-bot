"""Sentiment scorer tests. We don't hit the Anthropic API anywhere — the real
client gets mocked, and the StubSentimentScorer is exercised on its own."""

import aiosqlite
import pytest

from src.features.sentiment import (
    ClaudeSentimentScorer,
    SentimentResult,
    StubSentimentScorer,
    _parse_sentiment_json,
    _ticker_for,
    get_default_scorer,
    score_unscored_news,
)


# ---------- _parse_sentiment_json ----------

def test_parse_clean_json():
    r = _parse_sentiment_json('{"score": 0.7, "confidence": 0.8, "decay_minutes": 30}', "haiku")
    assert r.score == 0.7
    assert r.confidence == 0.8
    assert r.decay_minutes == 30
    assert r.model == "haiku"


def test_parse_strips_code_fence():
    raw = '```json\n{"score": -0.5, "confidence": 0.9, "decay_minutes": 60}\n```'
    r = _parse_sentiment_json(raw, "haiku")
    assert r.score == -0.5
    assert r.decay_minutes == 60


def test_parse_clamps_out_of_range_scores():
    """A misbehaving model returning 5.0 must not poison features downstream."""
    r = _parse_sentiment_json('{"score": 5.0, "confidence": -2.0, "decay_minutes": 9999}', "haiku")
    assert r.score == 1.0
    assert r.confidence == 0.0
    assert 0 <= r.decay_minutes <= 720


def test_parse_invalid_json_returns_neutral():
    r = _parse_sentiment_json("this is not json", "haiku")
    assert r.score == 0.0
    assert r.confidence == 0.0
    assert "unparseable" in r.model


def test_parse_missing_keys_returns_neutral():
    r = _parse_sentiment_json('{"score": 0.5}', "haiku")
    assert r.score == 0.0
    assert "invalid" in r.model


def test_parse_non_numeric_returns_neutral():
    r = _parse_sentiment_json(
        '{"score": "very positive", "confidence": "high", "decay_minutes": "soon"}', "haiku"
    )
    assert r.score == 0.0
    assert "invalid" in r.model


# ---------- ticker extraction ----------

def test_ticker_for_extracts_isin_suffix():
    assert _ticker_for("NSE_EQ|INE002A01018") == "INE002A01018"


def test_ticker_for_none_for_empty():
    assert _ticker_for(None) is None
    assert _ticker_for("") is None


def test_ticker_for_passthrough_when_no_pipe():
    assert _ticker_for("RELIANCE") == "RELIANCE"


# ---------- Stub ----------

@pytest.mark.asyncio
async def test_stub_returns_neutral():
    s = StubSentimentScorer()
    r = await s.score("RELIANCE reports record quarterly profit", ticker="RELIANCE")
    assert r == SentimentResult(score=0.0, confidence=0.0, decay_minutes=0, model="stub")


# ---------- get_default_scorer ----------

def test_default_scorer_uses_stub_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = get_default_scorer()
    assert isinstance(s, StubSentimentScorer)


def test_default_scorer_uses_claude_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-not-real")
    s = get_default_scorer()
    assert isinstance(s, ClaudeSentimentScorer)


# ---------- Claude scorer with mocked client ----------

class _FakeTextBlock:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


@pytest.mark.asyncio
async def test_claude_scorer_parses_happy_path(mocker):
    scorer = ClaudeSentimentScorer(api_key="sk-test", model="haiku-test")
    mocker.patch.object(
        scorer.client.messages,
        "create",
        new_callable=mocker.AsyncMock,
        return_value=_FakeResponse('{"score": 0.8, "confidence": 0.9, "decay_minutes": 60}'),
    )
    r = await scorer.score("Reliance posts record Q4 profit", ticker="RELIANCE")
    assert r.score == 0.8
    assert r.confidence == 0.9
    assert r.decay_minutes == 60
    assert r.model == "haiku-test"


@pytest.mark.asyncio
async def test_claude_scorer_handles_api_exception(mocker):
    scorer = ClaudeSentimentScorer(api_key="sk-test", model="haiku-test")
    mocker.patch.object(
        scorer.client.messages,
        "create",
        side_effect=RuntimeError("simulated rate limit"),
    )
    r = await scorer.score("Whatever", ticker=None)
    assert r.score == 0.0
    assert r.confidence == 0.0
    assert "error" in r.model


@pytest.mark.asyncio
async def test_claude_scorer_sends_cached_system_prompt(mocker):
    scorer = ClaudeSentimentScorer(api_key="sk-test", model="haiku-test")
    create = mocker.patch.object(
        scorer.client.messages,
        "create",
        new_callable=mocker.AsyncMock,
        return_value=_FakeResponse('{"score": 0, "confidence": 0, "decay_minutes": 0}'),
    )
    await scorer.score("h", ticker="T")
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "haiku-test"
    # System prompt is a list-of-blocks with cache_control set — this is what unlocks prompt caching.
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0].get("cache_control") == {"type": "ephemeral"}
    # The actual rubric must be in the prompt for Claude to score consistently.
    assert "Indian equity" in kwargs["system"][0]["text"]


# ---------- Batch driver ----------

@pytest.fixture
async def news_db(tmp_path):
    """Minimal news table with the sentiment columns, populated with three rows."""
    db_path = str(tmp_path / "news.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                published_at INTEGER NOT NULL,
                fetched_at INTEGER NOT NULL,
                source TEXT NOT NULL,
                headline TEXT NOT NULL,
                url TEXT,
                instrument_key TEXT,
                sentiment_score REAL,
                sentiment_confidence REAL,
                sentiment_decay_minutes INTEGER,
                sentiment_model TEXT,
                sentiment_at INTEGER,
                UNIQUE(source, url)
            )
            """
        )
        await db.executemany(
            "INSERT INTO news (published_at, fetched_at, source, headline, url, instrument_key) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, 2, "et", "Nifty hits new high", "u1", None),
                (1, 2, "et", "RBI surprise rate cut", "u2", None),
                (1, 2, "google_news:RELIANCE", "Reliance Q4 beats estimates",
                 "u3", "NSE_EQ|INE002A01018"),
            ],
        )
        await db.commit()
    return db_path


@pytest.mark.asyncio
async def test_score_unscored_news_marks_every_row(news_db):
    scorer = StubSentimentScorer()
    n = await score_unscored_news(scorer, limit=10, db_path=news_db)
    assert n == 3
    async with aiosqlite.connect(news_db) as db:
        async with db.execute(
            "SELECT id, sentiment_at, sentiment_model FROM news ORDER BY id"
        ) as cur:
            rows = await cur.fetchall()
    for nid, sentiment_at, model in rows:
        assert sentiment_at is not None
        assert model == "stub"


@pytest.mark.asyncio
async def test_score_unscored_news_is_idempotent(news_db):
    """Second call should score 0 new rows because everything is already scored."""
    scorer = StubSentimentScorer()
    n1 = await score_unscored_news(scorer, limit=10, db_path=news_db)
    n2 = await score_unscored_news(scorer, limit=10, db_path=news_db)
    assert n1 == 3
    assert n2 == 0


@pytest.mark.asyncio
async def test_score_unscored_news_respects_limit(news_db):
    scorer = StubSentimentScorer()
    n = await score_unscored_news(scorer, limit=2, db_path=news_db)
    assert n == 2
    async with aiosqlite.connect(news_db) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM news WHERE sentiment_at IS NULL"
        ) as cur:
            (remaining,) = await cur.fetchone()
    assert remaining == 1


@pytest.mark.asyncio
async def test_score_unscored_news_passes_ticker_to_scorer(news_db, mocker):
    """The ticker derived from instrument_key must reach the scorer's score() call."""
    calls = []

    class _RecordingScorer:
        model_name = "rec"
        async def score(self, headline, ticker=None):
            calls.append((headline, ticker))
            return SentimentResult(0.0, 0.0, 0, "rec")

    await score_unscored_news(_RecordingScorer(), limit=10, db_path=news_db)
    # The RELIANCE row should have come through with the ISIN-suffix ticker.
    tickers = [t for _, t in calls]
    assert "INE002A01018" in tickers
    # Market-wide rows pass ticker=None.
    assert None in tickers
