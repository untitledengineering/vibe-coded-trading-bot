// ============ DOM handles ============
const $ = (id) => document.getElementById(id);

const istClockEl      = $("ist-clock");
const marketStatusEl  = $("market-status");
const pnlHeroEl       = $("pnl-hero");
const pnlDeltaEl      = $("pnl-delta");
const tradesCountEl   = $("trades-count");
const winLossEl       = $("win-loss");
const openCountEl     = $("open-count");
const costsEstEl      = $("costs-est");
const sparkSvg        = $("spark-svg");
const sparkPath       = $("spark-path");
const btnHalt         = $("btn-halt");
const btnResume       = $("btn-resume");
const btnReport       = $("btn-report");
const orbEngine       = $("orb-engine");
const engineDetail    = $("engine-detail");
const orbStream       = $("orb-stream");
const streamDetail    = $("stream-detail");
const orbAuth         = $("orb-auth");
const authDetail      = $("auth-detail");
const btnConnect      = $("btn-connect");
const btnLogout       = $("btn-logout");
const positionsGrid   = $("positions-grid");
const openCountBadge  = $("open-count-badge");
const tradesList      = $("trades-list");
const tradesCountBadge= $("trades-count-badge");
const footerModel     = $("footer-model");
const footerCycles    = $("footer-cycles");
const haltBanner      = $("halt-banner");
const haltReasonEl    = $("halt-reason");

// ============ Tab routing ============
let activeTab = "live";

function switchTab(name) {
    activeTab = name;
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === name);
    });
    document.querySelectorAll(".tab-content").forEach(el => {
        el.classList.toggle("hidden", el.id !== `tab-${name}`);
    });
    // Trigger immediate fetch for the activated tab
    if (name === "scanner") fetchScanner();
    if (name === "news") fetchNews();
    if (name === "signals") fetchSignals();
}

document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// ============ Scanner window selector ============
let scannerWindow = 15;
document.querySelectorAll(".window-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".window-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        scannerWindow = parseInt(btn.dataset.window, 10);
        fetchScanner();
    });
});

// ============ Live LTPs via SSE ============
const liveQuotes = {};
let eventSource = null;

function startSSE() {
    if (eventSource) return;
    eventSource = new EventSource("/stream");
    eventSource.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            if (data && data.feeds) {
                for (const [key, payload] of Object.entries(data.feeds)) {
                    const ltp = payload?.ltpc?.ltp ?? payload?.ff?.marketFF?.ltpc?.ltp;
                    if (typeof ltp === "number") liveQuotes[key] = ltp;
                }
            }
        } catch (_) {}
        // Refresh position cards in-place using cached paper data
        if (cachedPaper && cachedPaper.open_positions && activeTab === "live") {
            updatePositionQuotes(cachedPaper.open_positions);
        }
    };
    eventSource.onerror = () => {
        eventSource.close();
        eventSource = null;
        setTimeout(startSSE, 3000);
    };
}

function stopSSE() {
    if (eventSource) { eventSource.close(); eventSource = null; }
}

// ============ Formatters ============
const inr = (v) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const abs = Math.abs(v).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return (v < 0 ? "-" : v > 0 ? "+" : "") + "₹" + abs;
};

const inrCompact = (v) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return (v > 0 ? "+" : "") + "₹" + v.toFixed(2);
};

const pct = (v) => {
    if (v === null || v === undefined) return "—";
    return (v > 0 ? "+" : "") + v.toFixed(3) + "%";
};

const fmtTimeIst = (epoch) => {
    if (!epoch) return "—";
    return new Date(epoch * 1000).toLocaleTimeString("en-IN", {
        hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Kolkata"
    });
};

const fmtDateTimeIst = (epoch) => {
    if (!epoch) return "—";
    return new Date(epoch * 1000).toLocaleString("en-IN", {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        hour12: false, timeZone: "Asia/Kolkata"
    });
};

const sentimentLabel = (score) => {
    if (score === null || score === undefined) return { text: "neutral", cls: "sent-neutral" };
    if (score > 0.3) return { text: `+${score.toFixed(2)} bullish`, cls: "sent-bull" };
    if (score < -0.3) return { text: `${score.toFixed(2)} bearish`, cls: "sent-bear" };
    return { text: `${score.toFixed(2)} neutral`, cls: "sent-neutral" };
};

// ============ Hero number tween ============
let pnlAnimFrame = null;
let pnlDisplayed = 0;

function tweenPnl(target) {
    if (pnlAnimFrame) cancelAnimationFrame(pnlAnimFrame);
    const start = pnlDisplayed, delta = target - start, dur = 400, t0 = performance.now();
    const step = (now) => {
        const t = Math.min(1, (now - t0) / dur);
        pnlDisplayed = start + delta * (1 - Math.pow(1 - t, 3));
        renderPnlText(pnlDisplayed, target);
        if (t < 1) pnlAnimFrame = requestAnimationFrame(step);
        else { pnlDisplayed = target; pnlAnimFrame = null; }
    };
    pnlAnimFrame = requestAnimationFrame(step);
}

function renderPnlText(displayed, finalTarget) {
    pnlHeroEl.textContent = inrCompact(displayed);
    pnlHeroEl.className = `pnl-big mono ${finalTarget > 0 ? "up" : finalTarget < 0 ? "down" : "flat"}`;
}

// ============ Sparkline ============
function renderSparkline(series, currentPnl) {
    if (!series || series.length === 0) {
        sparkPath.setAttribute("d", "");
        sparkSvg.className = "flat";
        return;
    }
    const W = 200, H = 50, PAD = 4;
    const ys = series.map(p => p.cum_pnl);
    const minY = Math.min(0, ...ys), maxY = Math.max(0, ...ys);
    const rangeY = Math.max(0.01, maxY - minY);
    const n = series.length, stepX = n === 1 ? 0 : (W - PAD * 2) / (n - 1);
    const points = series.map((p, i) => {
        const x = PAD + i * stepX;
        const y = H - PAD - ((p.cum_pnl - minY) / rangeY) * (H - PAD * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    sparkPath.setAttribute("d", "M" + points.join(" L"));
    sparkSvg.className = currentPnl > 0 ? "up" : currentPnl < 0 ? "down" : "flat";
}

// ============ Position card render ============
function rationale(p) {
    const parts = [];
    if (p.predicted_return != null) {
        const sign = p.predicted_return > 0 ? "+" : "";
        parts.push(`model ${sign}${(p.predicted_return * 100).toFixed(2)}%`);
    }
    if (p.entry_sentiment_score != null) {
        const sl = sentimentLabel(p.entry_sentiment_score);
        parts.push(`sent ${p.entry_sentiment_score > 0 ? "+" : ""}${p.entry_sentiment_score.toFixed(2)}`);
    }
    return parts.length ? parts.join(" · ") : null;
}

function renderPositions(positions) {
    openCountBadge.textContent = positions.length;
    openCountEl.textContent = positions.length;
    if (positions.length === 0) {
        positionsGrid.innerHTML = '<div class="empty">No open positions.</div>';
        return;
    }
    positionsGrid.innerHTML = "";
    for (const p of positions) {
        const card = document.createElement("div");
        card.className = "position-card";
        card.dataset.key = p.instrument_key;

        const quote = liveQuotes[p.instrument_key] ?? p.last_quote;
        const pnlVal = quote != null
            ? p.qty * (p.side === "long" ? (quote - p.entry_price) : (p.entry_price - quote))
            : (p.unrealised_pnl_inr ?? 0);

        const sl = p.stop_loss_price, tp = p.target_price;
        const range = Math.abs(tp - sl);
        let markerPct = 50;
        if (quote != null && range > 0) {
            markerPct = p.side === "long"
                ? ((quote - sl) / range) * 100
                : ((sl - quote) / range) * 100;
            markerPct = Math.max(0, Math.min(100, markerPct));
        }

        const rat = rationale(p);
        card.innerHTML = `
            <div class="pos-row">
                <span class="pos-symbol">${p.trading_symbol}</span>
                <span class="pos-side ${p.side}">${p.side}</span>
            </div>
            <div class="pos-row">
                <span class="pos-prices">qty ${p.qty} · @${p.entry_price.toFixed(2)} → ${quote != null ? quote.toFixed(2) : "—"}</span>
                <span class="pos-time mono">${fmtTimeIst(p.entry_ts)}</span>
            </div>
            ${rat ? `<div class="pos-rationale">${rat}</div>` : ""}
            <div class="pos-pnl ${pnlVal > 0 ? "up" : pnlVal < 0 ? "down" : "flat"}" data-pnl="${p.instrument_key}">${inrCompact(pnlVal)}</div>
            <div class="sltp-bar">
                <div class="sltp-marker" style="left: calc(${markerPct}% - 6px);"></div>
            </div>
            <div class="sltp-labels">
                <span>SL ${sl.toFixed(2)}</span>
                <span>TP ${tp.toFixed(2)}</span>
            </div>
        `;
        positionsGrid.appendChild(card);
    }
}

function updatePositionQuotes(positions) {
    for (const p of positions) {
        const quote = liveQuotes[p.instrument_key];
        if (quote == null) continue;
        const pnlEl = positionsGrid.querySelector(`[data-pnl="${p.instrument_key}"]`);
        if (!pnlEl) continue;
        const pnlVal = p.qty * (p.side === "long" ? (quote - p.entry_price) : (p.entry_price - quote));
        pnlEl.textContent = inrCompact(pnlVal);
        pnlEl.className = `pos-pnl ${pnlVal > 0 ? "up" : pnlVal < 0 ? "down" : "flat"}`;
    }
}

// ============ Trades list render ============
function renderTrades(trades) {
    tradesCountBadge.textContent = trades.length;
    tradesCountEl.textContent = trades.length;
    if (trades.length === 0) {
        tradesList.innerHTML = '<div class="empty">No trades closed today.</div>';
        return;
    }
    tradesList.innerHTML = "";
    for (const t of trades) {
        const row = document.createElement("div");
        row.className = "trade-row";
        const pnlVal = t.realised_pnl_inr ?? 0;
        const pnlClass = pnlVal > 0 ? "up" : pnlVal < 0 ? "down" : "";
        const rat = rationale(t);
        row.innerHTML = `
            <div class="trade-main">
                <span class="trade-time mono">${fmtTimeIst(t.entry_ts)} → ${fmtTimeIst(t.exit_ts)}</span>
                <span class="trade-symbol">${t.trading_symbol}</span>
                <span class="trade-side ${t.side}">${t.side}</span>
                <span class="mono trade-px">@${t.entry_price.toFixed(2)} → ${t.exit_price != null ? t.exit_price.toFixed(2) : "—"}</span>
                <span class="trade-reason">${t.exit_reason ?? ""}</span>
                <span class="trade-pnl mono ${pnlClass}">${inrCompact(pnlVal)}</span>
            </div>
            ${rat ? `<div class="trade-rationale">${rat}</div>` : ""}
        `;
        tradesList.appendChild(row);
    }
}

// ============ Clock + market countdown ============
function updateClock(paperStatus) {
    const now = new Date();
    istClockEl.textContent = now.toLocaleTimeString("en-IN", {
        hour: "2-digit", minute: "2-digit", second: "2-digit",
        hour12: false, timeZone: "Asia/Kolkata"
    }) + " IST";

    if (paperStatus && paperStatus.market_open) {
        const secs = paperStatus.seconds_to_market_close;
        if (typeof secs === "number" && secs > 0) {
            const h = Math.floor(secs / 3600), m = Math.floor((secs % 3600) / 60);
            marketStatusEl.textContent = `closes in ${h}h ${m}m`;
            marketStatusEl.className = "";
        } else {
            marketStatusEl.textContent = "closing";
            marketStatusEl.className = "muted";
        }
    } else {
        const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
        const day = ist.getDay();
        let secsUntilOpen = (9 * 60 + 15) * 60 - (ist.getHours() * 3600 + ist.getMinutes() * 60 + ist.getSeconds());
        if (secsUntilOpen <= 0 || day === 0 || day === 6) {
            let ahead = secsUntilOpen <= 0 ? 1 : 0;
            let next = (day + ahead) % 7;
            while (next === 0 || next === 6) { ahead++; next = (day + ahead) % 7; }
            secsUntilOpen += ahead * 86400;
        }
        if (secsUntilOpen > 0 && secsUntilOpen < 86400 * 4) {
            const h = Math.floor(secsUntilOpen / 3600), m = Math.floor((secsUntilOpen % 3600) / 60);
            marketStatusEl.textContent = `opens in ${h}h ${m}m`;
        } else {
            marketStatusEl.textContent = "market closed";
        }
        marketStatusEl.className = "muted";
    }
}

// ============ Status orb renders ============
function renderAuth(auth) {
    if (auth && auth.authenticated) {
        orbAuth.className = "orb ok";
        const secs = auth.seconds_until_expiry ?? 0;
        authDetail.textContent = `valid for ${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
        btnConnect.classList.add("hidden");
        btnLogout.classList.remove("hidden");
    } else {
        orbAuth.className = "orb err";
        authDetail.textContent = "not connected";
        btnConnect.classList.remove("hidden");
        btnLogout.classList.add("hidden");
    }
}

function renderStream(health) {
    if (!health) { orbStream.className = "orb"; streamDetail.textContent = "—"; return; }
    if (!health.streamer_running) { orbStream.className = "orb"; streamDetail.textContent = "idle"; return; }
    if (health.stale) { orbStream.className = "orb warn"; streamDetail.textContent = `stalled (${health.last_tick_seconds_ago}s)`; return; }
    const ago = health.last_tick_seconds_ago;
    if (!health.market_open) {
        orbStream.className = "orb";
        streamDetail.textContent = ago != null && ago < 120 ? "snapshot" : "idle (market closed)";
    } else if (ago == null) {
        orbStream.className = "orb warn"; streamDetail.textContent = "warming up";
    } else {
        orbStream.className = "orb ok"; streamDetail.textContent = `live · ${ago}s ago`;
    }
}

// ============ Paper status render ============
function renderPaper(paper) {
    if (!paper || paper.running === false) {
        orbEngine.className = "orb"; engineDetail.textContent = "not running";
        haltBanner.classList.add("hidden"); return;
    }
    if (paper.halted) {
        orbEngine.className = "orb err"; engineDetail.textContent = "halted";
        haltBanner.classList.remove("hidden");
        haltReasonEl.textContent = paper.halt_reason || "manual";
        btnHalt.classList.add("hidden"); btnResume.classList.remove("hidden");
    } else {
        orbEngine.className = "orb ok";
        engineDetail.textContent = (paper.market_open ?? false) ? `live · ${paper.model}` : `waiting · ${paper.model}`;
        haltBanner.classList.add("hidden");
        btnHalt.classList.remove("hidden"); btnResume.classList.add("hidden");
    }

    // Hero
    const net = paper.net_pnl_inr ?? 0;
    tweenPnl(net);
    pnlDeltaEl.textContent = `realised ${inrCompact(paper.realised_pnl_inr ?? 0)} · open ${inrCompact(paper.unrealised_pnl_inr ?? 0)}`;
    winLossEl.textContent = `${paper.wins_today ?? 0}W / ${paper.losses_today ?? 0}L`;
    renderSparkline(paper.pnl_series, net);

    // Estimate costs from gross - net for each closed trade
    let totalCosts = 0;
    for (const t of (paper.closed_trades_today || [])) {
        if (t.entry_price && t.exit_price && t.realised_pnl_inr != null) {
            const gross = t.qty * (t.side === "long"
                ? (t.exit_price - t.entry_price)
                : (t.entry_price - t.exit_price));
            totalCosts += gross - t.realised_pnl_inr;
        }
    }
    costsEstEl.textContent = totalCosts !== 0 ? inrCompact(-Math.abs(totalCosts)) : "—";

    // Live tab
    if (activeTab === "live") renderPositions(paper.open_positions || []);

    // Trades tab
    renderTrades(paper.closed_trades_today || []);

    // Risk tab
    renderRisk(paper);

    footerModel.textContent = paper.model || "—";
    footerCycles.textContent = paper.cycles_run ?? "—";
}

// ============ Risk tab render ============
function renderRisk(paper) {
    $("risk-model").textContent = paper.model || "—";
    $("risk-cycles").textContent = paper.cycles_run ?? "—";
    const trades = paper.closed_trades_today || [];
    $("risk-trades").textContent = trades.length;
    const wins = paper.wins_today ?? 0;
    const losses = paper.losses_today ?? 0;
    const total = wins + losses;
    $("risk-winrate").textContent = total > 0 ? `${(wins / total * 100).toFixed(1)}%` : "—";

    const netPnl = paper.realised_pnl_inr ?? 0;
    $("risk-avg-pnl").textContent = trades.length > 0 ? inrCompact(netPnl / trades.length) : "—";

    let best = -Infinity, worst = Infinity;
    let gross = 0, costs = 0;
    let slExits = 0, tpExits = 0, eodExits = 0;
    const symbols = new Set();
    for (const t of trades) {
        const p = t.realised_pnl_inr ?? 0;
        if (p > best) best = p;
        if (p < worst) worst = p;
        if (t.entry_price && t.exit_price) {
            const g = t.qty * (t.side === "long"
                ? (t.exit_price - t.entry_price)
                : (t.entry_price - t.exit_price));
            gross += g;
            costs += g - p;
        }
        symbols.add(t.instrument_key);
        if (t.exit_reason === "stop_loss") slExits++;
        else if (t.exit_reason === "target") tpExits++;
        else if (t.exit_reason === "eod") eodExits++;
    }

    $("risk-best").textContent = trades.length > 0 ? inrCompact(best) : "—";
    $("risk-worst").textContent = trades.length > 0 ? inrCompact(worst) : "—";
    $("risk-gross").textContent = trades.length > 0 ? inrCompact(gross) : "—";
    $("risk-net").textContent = inrCompact(netPnl);
    $("risk-costs").textContent = trades.length > 0 ? inrCompact(-Math.abs(costs)) : "—";
    $("risk-open").textContent = (paper.open_positions || []).length;
    $("risk-symbols").textContent = symbols.size;
    $("risk-sl-exits").textContent = slExits;
    $("risk-tp-exits").textContent = tpExits;
    $("risk-eod-exits").textContent = eodExits;

    // Skip reasons
    const skips = paper.last_skip_reasons || {};
    const skipEl = $("risk-skips");
    const entries = Object.entries(skips).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) {
        skipEl.innerHTML = '<div class="empty" style="padding:1rem;">No skips recorded.</div>';
    } else {
        skipEl.innerHTML = entries.map(([k, v]) =>
            `<div class="risk-row"><span>${k.replace(/_/g, " ")}</span><span class="mono">${v}</span></div>`
        ).join("");
    }
}

// ============ Scanner tab render ============
async function fetchScanner() {
    const el = $("gainers-list"), el2 = $("losers-list");
    try {
        const r = await fetch(`/paper/scanner?window=${scannerWindow}`);
        if (!r.ok) throw new Error(r.status);
        const data = await r.json();
        $("scanner-updated").textContent = `as of ${fmtTimeIst(data.computed_at)} (${data.total_symbols} symbols)`;
        renderScannerList(el, data.gainers, true);
        renderScannerList(el2, data.losers, false);
    } catch (e) {
        el.innerHTML = '<div class="empty">Failed to load.</div>';
        el2.innerHTML = '<div class="empty">Failed to load.</div>';
    }
}

function renderScannerList(container, items, isGainer) {
    if (!items || items.length === 0) {
        container.innerHTML = '<div class="empty">No data.</div>';
        return;
    }
    container.innerHTML = "";
    for (const item of items) {
        const row = document.createElement("div");
        row.className = "scanner-row";
        const retClass = isGainer ? "up" : "down";
        const signal = item.model_signal;
        let signalBadge = "";
        if (signal != null) {
            const sClass = signal > 0 ? "sig-bull" : signal < 0 ? "sig-bear" : "";
            signalBadge = `<span class="sig-badge ${sClass}">${signal > 0 ? "▲" : "▼"} ${(Math.abs(signal) * 100).toFixed(2)}%</span>`;
        }
        const sent = item.sentiment_score;
        let sentBadge = "";
        if (sent != null) {
            const sl = sentimentLabel(sent);
            sentBadge = `<span class="sent-badge ${sl.cls}">${sl.text}</span>`;
        }
        row.innerHTML = `
            <span class="scanner-symbol">${item.trading_symbol}</span>
            <span class="scanner-ret mono ${retClass}">${pct(item.return_pct)}</span>
            <span class="scanner-price mono muted">${item.ltp != null ? item.ltp.toFixed(2) : item.close_now}</span>
            <span class="scanner-badges">${signalBadge}${sentBadge}</span>
        `;
        container.appendChild(row);
    }
}

// ============ News tab render ============
async function fetchNews() {
    const el = $("news-list");
    try {
        const r = await fetch("/paper/news?hours=6");
        if (!r.ok) throw new Error(r.status);
        const data = await r.json();
        $("news-updated").textContent = `${data.total} headlines · last 6h`;

        // Market sentiment badge
        const ms = data.market_sentiment;
        const msBadge = $("market-sentiment-badge");
        if (ms != null) {
            const sl = sentimentLabel(ms);
            msBadge.textContent = `market ${sl.text}`;
            msBadge.className = `badge ${sl.cls}`;
        } else {
            msBadge.textContent = "market sentiment —";
            msBadge.className = "badge muted";
        }

        if (!data.news || data.news.length === 0) {
            el.innerHTML = '<div class="empty">No recent scored news.</div>';
            return;
        }
        el.innerHTML = "";
        for (const n of data.news) {
            const row = document.createElement("div");
            row.className = "news-row";
            const sl = n.sentiment_score != null ? sentimentLabel(n.sentiment_score) : null;
            const sentHtml = sl
                ? `<span class="sent-badge ${sl.cls}">${sl.text}</span>`
                : `<span class="sent-badge sent-neutral">unscored</span>`;
            const link = n.url
                ? `<a href="${n.url}" target="_blank" rel="noopener" class="news-link">${n.headline}</a>`
                : `<span>${n.headline}</span>`;
            row.innerHTML = `
                <div class="news-meta">
                    <span class="news-time mono muted">${fmtTimeIst(n.published_at)}</span>
                    <span class="news-source muted">${n.source}</span>
                    ${n.trading_symbol ? `<span class="news-ticker">${n.trading_symbol}</span>` : ""}
                    ${sentHtml}
                </div>
                <div class="news-headline">${link}</div>
            `;
            el.appendChild(row);
        }
    } catch (e) {
        el.innerHTML = '<div class="empty">Failed to load news.</div>';
    }
}

// ============ Signals tab render ============
async function fetchSignals() {
    try {
        const r = await fetch("/paper/signals?top_n=30");
        if (!r.ok) throw new Error(r.status);
        const data = await r.json();

        $("signals-model-badge").textContent = data.model ? `model · ${data.model}` : "—";
        $("signals-updated").textContent = data.computed_at
            ? `as of ${fmtTimeIst(data.computed_at)}`
            : "not yet computed";

        renderSignalsList($("signals-bullish"), data.bullish || [], true);
        renderSignalsList($("signals-bearish"), data.bearish || [], false);
    } catch (e) {
        $("signals-bullish").innerHTML = '<div class="empty">Failed to load.</div>';
        $("signals-bearish").innerHTML = '<div class="empty">Failed to load.</div>';
    }
}

function renderSignalsList(container, items, bullish) {
    if (!items || items.length === 0) {
        container.innerHTML = '<div class="empty">No signals.</div>';
        return;
    }
    container.innerHTML = "";
    for (const s of items) {
        const row = document.createElement("div");
        row.className = "signal-row";
        const retClass = bullish ? "up" : "down";
        const sentHtml = s.sentiment_score != null
            ? `<span class="sent-badge ${sentimentLabel(s.sentiment_score).cls}">${s.sentiment_score > 0 ? "+" : ""}${s.sentiment_score.toFixed(2)}</span>`
            : "";
        row.innerHTML = `
            <span class="signal-symbol">${s.trading_symbol}</span>
            <span class="signal-ret mono ${retClass}">${(s.predicted_return * 100).toFixed(3)}%</span>
            <span class="signal-ltp mono muted">${s.ltp != null ? s.ltp.toFixed(2) : "—"}</span>
            ${sentHtml}
        `;
        container.appendChild(row);
    }
}

// ============ Buttons ============
btnHalt.addEventListener("click", async () => {
    if (!confirm("Halt new entries for the rest of today?")) return;
    btnHalt.disabled = true;
    try { await fetch("/paper/halt", { method: "POST" }); await poll(); } finally { btnHalt.disabled = false; }
});

btnResume.addEventListener("click", async () => {
    btnResume.disabled = true;
    try { await fetch("/paper/resume", { method: "POST" }); await poll(); } finally { btnResume.disabled = false; }
});

btnReport.addEventListener("click", async () => {
    btnReport.disabled = true;
    try {
        const r = await fetch("/paper/report", { method: "POST" });
        const data = await r.json();
        if (data.path) alert(`Report written to:\n${data.path}`);
    } finally { btnReport.disabled = false; }
});

btnLogout.addEventListener("click", async () => {
    if (!confirm("Disconnect from Upstox and clear the stored token?")) return;
    btnLogout.disabled = true;
    try { await fetch("/auth/logout", { method: "POST" }); await poll(); } finally { btnLogout.disabled = false; }
});

// ============ Poll loop ============
let cachedPaper = null;

async function poll() {
    let auth = null, health = null, paper = null;
    try {
        [auth, health, paper] = await Promise.all([
            fetch("/auth/status").then(r => r.ok ? r.json() : null).catch(() => null),
            fetch("/streamer/health").then(r => r.ok ? r.json() : null).catch(() => null),
            fetch("/paper/status").then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
    } catch (_) {}

    renderAuth(auth);
    renderStream(health);
    if (paper) { renderPaper(paper); cachedPaper = paper; }
    if (auth && auth.authenticated) startSSE(); else stopSSE();
}

// Per-tab slow polls (scanner/news/signals refresh at their own pace)
let scannerTimer = null, newsTimer = null, signalsTimer = null;

function startTabPolls() {
    clearInterval(scannerTimer); clearInterval(newsTimer); clearInterval(signalsTimer);
    scannerTimer = setInterval(() => { if (activeTab === "scanner") fetchScanner(); }, 30_000);
    newsTimer    = setInterval(() => { if (activeTab === "news")    fetchNews();    }, 60_000);
    signalsTimer = setInterval(() => { if (activeTab === "signals") fetchSignals(); }, 30_000);
}

function tick() { updateClock(cachedPaper); }

poll();
tick();
setInterval(poll, 3000);
setInterval(tick, 1000);
startTabPolls();
