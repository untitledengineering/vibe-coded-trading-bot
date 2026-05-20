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
    if (name === "market") fetchMovers();
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
        // Patch live LTPs into market mover cards for streamed symbols
        updateMoverQuotes();
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
        const estCost = p.estimated_cost_inr ?? 0;
        const cleared = pnlVal > estCost;
        const costLabel = estCost > 0
            ? (cleared
                ? `cost ₹${estCost.toFixed(0)} · <span style="color:var(--green)">cleared ✓</span>`
                : `cost ₹${estCost.toFixed(0)} · <span style="color:var(--muted)">need ₹${(estCost - pnlVal).toFixed(0)} more</span>`)
            : "";
        card.innerHTML = `
            <div class="pos-row">
                <span class="pos-symbol" data-sym-key="${p.instrument_key}">${displaySymbol(p.trading_symbol, p.instrument_key)}</span>
                <span class="pos-side ${p.side}">${p.side}</span>
                <button class="btn btn-small btn-warn btn-close-pos" data-key="${p.instrument_key}" data-symbol="${p.trading_symbol}" style="margin-left:auto;">Close</button>
            </div>
            <div class="pos-row">
                <span class="pos-prices" data-price="${p.instrument_key}">qty ${p.qty} · @${p.entry_price.toFixed(2)} → ${quote != null ? quote.toFixed(2) : "—"}</span>
                <span class="pos-time mono">${fmtTimeIst(p.entry_ts)}</span>
            </div>
            ${rat ? `<div class="pos-rationale">${rat}</div>` : ""}
            <div class="pos-pnl ${pnlVal > 0 ? "up" : pnlVal < 0 ? "down" : "flat"}" data-pnl="${p.instrument_key}">${inrCompact(pnlVal)}</div>
            ${costLabel ? `<div class="pos-cost-row">${costLabel}</div>` : ""}
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
        const pnlVal = p.qty * (p.side === "long" ? (quote - p.entry_price) : (p.entry_price - quote));

        const pnlEl = positionsGrid.querySelector(`[data-pnl="${p.instrument_key}"]`);
        if (pnlEl) {
            pnlEl.textContent = inrCompact(pnlVal);
            pnlEl.className = `pos-pnl ${pnlVal > 0 ? "up" : pnlVal < 0 ? "down" : "flat"}`;
        }
        const priceEl = positionsGrid.querySelector(`[data-price="${p.instrument_key}"]`);
        if (priceEl) {
            priceEl.textContent = `qty ${p.qty} · @${p.entry_price.toFixed(2)} → ${quote.toFixed(2)}`;
        }
    }
}

// ============ Fast quote poller (2s) for open positions ============
let _quotePollTimer = null;

async function pollPositionQuotes() {
    if (!cachedPaper || !(cachedPaper.open_positions || []).length) return;
    try {
        const r = await fetch("/paper/quotes");
        if (!r.ok) return;
        const quotes = await r.json();
        let changed = false;
        for (const [key, ltp] of Object.entries(quotes)) {
            if (liveQuotes[key] !== ltp) { liveQuotes[key] = ltp; changed = true; }
        }
        if (changed && cachedPaper && activeTab === "live") {
            updatePositionQuotes(cachedPaper.open_positions || []);
        }
    } catch (_) {}
}

function startQuotePoll() {
    if (_quotePollTimer) return;
    _quotePollTimer = setInterval(pollPositionQuotes, 1000);
}
function stopQuotePoll() {
    clearInterval(_quotePollTimer);
    _quotePollTimer = null;
}

// ============ Symbol display helpers ============
const _symbolCache = {};

function displaySymbol(sym, key) {
    // If the API returned the raw instrument key (not in universe), show the
    // exchange-stripped version immediately and kick off an async resolution.
    if (!sym || sym.includes("|")) {
        const short = sym ? sym.split("|")[1] : key.split("|")[1];
        if (!_symbolCache[key]) resolveSymbolAsync(key);
        return _symbolCache[key] || short;
    }
    return sym;
}

async function resolveSymbolAsync(key) {
    if (_symbolCache[key]) return;
    try {
        const r = await fetch(`/stock/${encodeURIComponent(key)}/info`);
        if (!r.ok) return;
        const d = await r.json();
        const name = d.trading_symbol || d.name;
        if (name && !name.includes("|")) {
            _symbolCache[key] = name;
            // Patch all visible elements that still show the raw key
            document.querySelectorAll(`[data-sym-key="${key}"]`).forEach(el => {
                el.textContent = name;
            });
        }
    } catch (_) {}
}

// ============ Activity panel ============
const _SKIP_LABELS = {
    below_edge_threshold: (n, paper) => `${n} signals below ${((paper?.min_predicted_edge ?? 0.003)*100).toFixed(2)}% edge · watching for conviction`,
    below_cost_floor:     n => `${n} signals too small to cover costs`,
    max_concurrent_reached: n => `all ${n > 1 ? n : 4} slots full`,
    no_features_yet:      () => `building bar data · first decision soon`,
    already_open:         n => `${n} symbols already held`,
    cooldown:             n => `${n} symbols in 30-min cooldown`,
    regime_suppressed:    n => `${n} signals suppressed by market regime`,
};

function updateActivityPanel(paper) {
    const dot    = document.getElementById("activity-dot");
    const cycle  = document.getElementById("activity-cycle");
    const stream = document.getElementById("activity-stream");
    const status = document.getElementById("activity-status");
    const sigs   = document.getElementById("activity-top-signals");
    if (!dot) return;

    const age = paper.last_cycle_at
        ? Math.round(paper.now_ts - paper.last_cycle_at)
        : null;
    const alive = age !== null && age < 10;
    dot.className = "activity-dot " + (alive ? "alive" : "stale");
    cycle.textContent = `cycle #${paper.cycles_run ?? "—"}${age !== null ? " · " + age + "s ago" : ""}`;

    // Stream freshness from streamer_health block (top-level health pill populates this)
    const sh = paper.streamer_health;
    if (sh) {
        const ta = sh.last_tick_seconds_ago;
        stream.textContent = ta != null ? `stream ${ta}s ago` : "stream —";
    }

    // Skip reason → human sentence
    const reasons = paper.last_skip_reasons || {};
    const lines = Object.entries(reasons).map(([k, n]) => {
        const fn = _SKIP_LABELS[k];
        return fn ? fn(n, paper) : `${k}: ${n}`;
    });
    const decisionAgo = paper.last_decision_ts
        ? Math.round((paper.now_ts - paper.last_decision_ts) / 60)
        : null;
    const decisionStr = decisionAgo !== null
        ? ` (last model run ${decisionAgo}m ago)`
        : "";
    status.textContent = lines.length
        ? lines.join(" · ") + decisionStr
        : paper.running ? "scanning…" + decisionStr : "engine stopped";

    // Top signals pills
    sigs.innerHTML = "";
    const top = paper.top_signals || [];
    for (const { k, p } of top.slice(0, 6)) {
        const sym = _symbolCache[k] || k.split("|")[1] || k;
        const pill = document.createElement("span");
        pill.className = `sig-pill ${p >= 0 ? "bull" : "bear"}`;
        pill.textContent = `${sym} ${p >= 0 ? "+" : ""}${(p * 100).toFixed(2)}%`;
        sigs.appendChild(pill);
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
        const costStr = t.actual_cost_inr != null ? `· cost ₹${t.actual_cost_inr.toFixed(0)}` : "";
        row.innerHTML = `
            <div class="trade-main">
                <span class="trade-time mono">${fmtTimeIst(t.entry_ts)} → ${fmtTimeIst(t.exit_ts)}</span>
                <span class="trade-symbol" data-sym-key="${t.instrument_key}">${displaySymbol(t.trading_symbol, t.instrument_key)}</span>
                <span class="trade-side ${t.side}">${t.side}</span>
                <span class="mono trade-px">@${t.entry_price.toFixed(2)} → ${t.exit_price != null ? t.exit_price.toFixed(2) : "—"}</span>
                <span class="trade-reason">${t.exit_reason ?? ""}</span>
                <span class="trade-pnl mono ${pnlClass}">${inrCompact(pnlVal)} <span class="muted" style="font-size:0.75em;font-weight:400;">${costStr}</span></span>
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
    if (health.stale) {
        orbStream.className = "orb err";
        streamDetail.textContent = "no live data · bot paused";
        return;
    }
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
    updateActivityPanel(paper);

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
        const windowLabel = data.early_session ? "since open" : `${data.window_minutes}m`;
        $("scanner-updated").textContent = `${windowLabel} · ${data.total_symbols} symbols · ${fmtTimeIst(data.computed_at)}`;
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
                    <span class="muted" style="margin-left:auto;font-size:0.72rem;opacity:0.6">tap for scripts →</span>
                </div>
                <div class="news-headline">${link}</div>
            `;
            row.addEventListener("click", function (e) {
                if (e.target.tagName === "A") return; // let article link open normally
                openNewsModal(n);
            });
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

// Close individual position — event delegation on the grid
positionsGrid.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-close-pos");
    if (!btn) return;
    const key = btn.dataset.key, sym = btn.dataset.symbol;
    if (!confirm(`Close ${sym} now at market price?`)) return;
    btn.disabled = true;
    btn.textContent = "…";
    try {
        const r = await fetch(`/paper/close/${encodeURIComponent(key)}`, { method: "POST" });
        const data = await r.json();
        if (data.ok) { await poll(); }
        else { alert(data.error || "Failed to close position"); btn.disabled = false; btn.textContent = "Close"; }
    } catch (_) { alert("Network error"); btn.disabled = false; btn.textContent = "Close"; }
});

// Extend loss cap buttons
async function extendLossCap(amount) {
    const btn = amount === 500 ? $("btn-extend-500") : $("btn-extend-1k");
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "…";
    try {
        const r = await fetch(`/paper/extend-loss-cap?amount=${amount}`, { method: "POST" });
        const data = await r.json();
        if (data.ok) {
            btn.textContent = `✓ Cap ₹${data.effective_cap_inr.toLocaleString("en-IN")}`;
            btn.style.color = "var(--green)";
            await poll();
            setTimeout(() => { btn.textContent = orig; btn.style.color = ""; btn.disabled = false; }, 3000);
            return;
        } else {
            alert(data.error || "Failed");
        }
    } catch (_) { alert("Network error"); }
    btn.textContent = orig;
    btn.style.color = "";
    btn.disabled = false;
}
$("btn-extend-500").addEventListener("click", () => extendLossCap(500));
$("btn-extend-1k").addEventListener("click",  () => extendLossCap(1000));

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
    if (auth && auth.authenticated) { startSSE(); startQuotePoll(); } else { stopSSE(); stopQuotePoll(); }
}

// ============ Market movers tab ============
async function fetchMovers() {
    const gainersEl = $("gainers-cards"), losersEl = $("losers-cards");
    gainersEl.innerHTML = '<div class="empty">Loading all NSE stocks…</div>';
    losersEl.innerHTML  = '<div class="empty">Loading all NSE stocks…</div>';
    try {
        const r = await fetch("/market/movers?limit=25");
        if (!r.ok) throw new Error(r.status);
        const data = await r.json();
        $("movers-meta").textContent = `${data.total_stocks} stocks · as of ${fmtTimeIst(data.computed_at)}`;
        renderMoverCards(gainersEl, data.gainers, true);
        renderMoverCards(losersEl,  data.losers,  false);
    } catch(e) {
        gainersEl.innerHTML = `<div class="empty">Failed: ${e.message}</div>`;
        losersEl.innerHTML  = '<div class="empty">—</div>';
    }
}

function renderMoverCards(container, items, isGainer) {
    if (!items || items.length === 0) { container.innerHTML = '<div class="empty">No data.</div>'; return; }
    container.innerHTML = "";
    for (const s of items) {
        const card = document.createElement("div");
        card.className = "mover-card";
        card.dataset.key = s.instrument_key;
        card.dataset.open = s.open;
        card.style.cursor = "pointer";
        card.addEventListener("click", (e) => {
            if (e.target.closest(".btn-place-order")) return;
            const url = `/static/stock.html?key=${encodeURIComponent(s.instrument_key)}&symbol=${encodeURIComponent(s.trading_symbol)}&name=${encodeURIComponent(s.name || s.trading_symbol)}`;
            window.open(url, "_blank");
        });
        const chgClass = isGainer ? "up" : "down";
        const sign = isGainer ? "+" : "";
        const low = s.low != null ? s.low.toFixed(2) : "—";
        const high = s.high != null ? s.high.toFixed(2) : "—";
        // Use live SSE quote if available (only for 209 streamed F&O stocks), else REST price
        const liveLtp = liveQuotes[s.instrument_key];
        const displayLtp = liveLtp ?? s.ltp;
        const livePct = s.open > 0 ? (displayLtp - s.open) / s.open * 100 : s.change_pct;
        const liveSign = livePct >= 0 ? "+" : "";
        const liveClass = livePct >= 0 ? "up" : "down";
        card.innerHTML = `
            <div class="mover-top">
                <span class="mover-symbol">${s.trading_symbol}</span>
                <span class="mover-name">${s.name || ""}</span>
                <span class="mover-chg ${liveClass}" data-chg="${s.instrument_key}">${liveSign}${livePct.toFixed(2)}%</span>
            </div>
            <div class="mover-row2">
                <span class="mover-ltp" data-ltp="${s.instrument_key}">₹${displayLtp.toFixed(2)}</span>
                <span class="mover-range">L ${low} · H ${high}</span>
            </div>
            <div class="mover-actions">
                <button class="btn-long  btn-place-order" data-key="${s.instrument_key}" data-side="long"  data-sym="${s.trading_symbol}">Long ▲</button>
                <button class="btn-short btn-place-order" data-key="${s.instrument_key}" data-side="short" data-sym="${s.trading_symbol}">Short ▼</button>
            </div>
        `;
        container.appendChild(card);
    }
}

// Called by the SSE handler to patch live LTPs into visible mover cards
function updateMoverQuotes() {
    if (activeTab !== "market") return;
    document.querySelectorAll(".mover-card[data-key]").forEach(card => {
        const key = card.dataset.key;
        const ltp = liveQuotes[key];
        if (ltp == null) return;
        const open = parseFloat(card.dataset.open);
        const ltpEl = card.querySelector(`[data-ltp="${key}"]`);
        const chgEl = card.querySelector(`[data-chg="${key}"]`);
        if (ltpEl) ltpEl.textContent = `₹${ltp.toFixed(2)}`;
        if (chgEl && open > 0) {
            const pct = (ltp - open) / open * 100;
            chgEl.textContent = `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
            chgEl.className = `mover-chg ${pct >= 0 ? "up" : "down"}`;
        }
    });
}

// Event delegation for order buttons
document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-place-order");
    if (!btn) return;
    const key = btn.dataset.key, side = btn.dataset.side, sym = btn.dataset.sym;
    if (!confirm(`Place ${side.toUpperCase()} on ${sym}?`)) return;
    const allBtns = btn.closest(".mover-card").querySelectorAll(".btn-place-order");
    allBtns.forEach(b => b.disabled = true);
    try {
        const r = await fetch(`/paper/force-entry?instrument_key=${encodeURIComponent(key)}&side=${side}`, { method: "POST" });
        const data = await r.json();
        if (data.ok) {
            btn.textContent = "✓ Placed";
            await poll();
        } else {
            alert(data.error || "Failed to place order");
            allBtns.forEach(b => b.disabled = false);
        }
    } catch(_) {
        alert("Network error");
        allBtns.forEach(b => b.disabled = false);
    }
});

$("btn-refresh-movers").addEventListener("click", fetchMovers);

// Per-tab slow polls (scanner/news/signals refresh at their own pace)
let scannerTimer = null, newsTimer = null, signalsTimer = null, moversTimer = null;

function startTabPolls() {
    clearInterval(scannerTimer); clearInterval(newsTimer); clearInterval(signalsTimer); clearInterval(moversTimer);
    scannerTimer = setInterval(() => { if (activeTab === "scanner") fetchScanner(); }, 30_000);
    newsTimer    = setInterval(() => { if (activeTab === "news")    fetchNews();    }, 60_000);
    signalsTimer = setInterval(() => { if (activeTab === "signals") fetchSignals(); }, 30_000);
    moversTimer  = setInterval(() => { if (activeTab === "market")  fetchMovers();  }, 60_000);
}

function tick() { updateClock(cachedPaper); }

poll();
tick();
setInterval(poll, 3000);
setInterval(tick, 1000);
startTabPolls();

// ============ News → Related scripts modal ============
function openNewsModal(n) {
    const articleEl = $("modal-article-section");
    const stocksEl  = $("modal-stocks-section");

    const sl = n.sentiment_score != null ? sentimentLabel(n.sentiment_score) : null;
    const sentHtml = sl ? `<span class="sent-badge ${sl.cls}">${sl.text}</span>` : "";
    const ticker = n.trading_symbol ? `<span class="news-ticker">${n.trading_symbol}</span>` : "";
    const headline = n.url
        ? `<a href="${n.url}" target="_blank" rel="noopener">${n.headline}</a>`
        : n.headline;
    articleEl.innerHTML = `
        <div class="modal-article">
            <div class="modal-article-meta">
                <span class="mono">${fmtTimeIst(n.published_at)}</span>
                <span>${n.source || ""}</span>
                ${ticker}${sentHtml}
            </div>
            <div class="modal-article-headline">${headline}</div>
        </div>`;

    stocksEl.innerHTML = `<div class="muted" style="font-size:0.85rem;padding:0.5rem 0">Loading related scripts…</div>`;
    $("news-modal").classList.remove("hidden");
    document.body.style.overflow = "hidden";

    fetchRelatedStocks(n, stocksEl);
}

const _NEWS_STOP = new Set([
    "the","a","an","and","or","of","in","at","to","is","are","was","were","be",
    "been","has","have","had","that","this","for","on","with","as","by","from",
    "its","it","new","will","says","said","after","but","up","down","over","may",
    "ltd","limited","pvt","private","india","indian","co","corp","industries",
    "reports","quarterly","results","profit","loss","revenue","q1","q2","q3","q4",
    "share","shares","stock","market","nse","bse","crore","lakh","rupees","rs",
]);

function _extractKeywords(headline) {
    return headline
        .replace(/[^a-zA-Z0-9 ]/g, " ")
        .split(/\s+/)
        .map(w => w.toLowerCase())
        .filter(w => w.length >= 3 && !_NEWS_STOP.has(w));
}

async function fetchRelatedStocks(n, el) {
    try {
        if (n.instrument_key) {
            // Directly linked stock — show it plus keyword search for more
            const [infoR, searchR] = await Promise.allSettled([
                fetch(`/stock/${encodeURIComponent(n.instrument_key)}/info`),
                fetch(`/market/search?q=${encodeURIComponent(n.headline)}&limit=3`),
            ]);
            let cards = "";
            let count = 0;
            if (infoR.status === "fulfilled" && infoR.value.ok) {
                const d = await infoR.value.json();
                cards += buildModalStockCard(d, n.instrument_key);
                count++;
            }
            if (searchR.status === "fulfilled" && searchR.value.ok) {
                const sd = await searchR.value.json();
                for (const r of (sd.results || [])) {
                    if (r.instrument_key === n.instrument_key) continue; // already shown
                    cards += buildModalStockCard({ trading_symbol: r.trading_symbol, name: r.name, ltp: null, change_pct: null }, r.instrument_key);
                    count++;
                }
            }
            if (!count) { el.innerHTML = '<div class="muted">Could not load stock data.</div>'; return; }
            el.innerHTML = `<div class="modal-stocks-title">Related scripts</div>` + cards;
        } else {
            // Market-wide news: search by headline keywords
            const kws = _extractKeywords(n.headline || "");
            if (!kws.length) { el.innerHTML = '<div class="muted">No specific stocks found for this headline.</div>'; return; }
            const query = kws.slice(0, 4).join(" ");
            const r = await fetch(`/market/search?q=${encodeURIComponent(query)}&limit=6`);
            if (!r.ok) { el.innerHTML = '<div class="muted">Search unavailable.</div>'; return; }
            const d = await r.json();
            if (!d.results || !d.results.length) {
                el.innerHTML = '<div class="muted">No matching scripts found for this article.</div>';
                return;
            }
            el.innerHTML = `<div class="modal-stocks-title">Mentioned in article</div>` +
                d.results.map(r => buildModalStockCard(
                    { trading_symbol: r.trading_symbol, name: r.name, ltp: null, change_pct: null },
                    r.instrument_key
                )).join("");
            // Async-fill prices for matched stocks
            for (const row of d.results) {
                fetch(`/stock/${encodeURIComponent(row.instrument_key)}/info`)
                    .then(res => res.ok ? res.json() : null)
                    .then(info => {
                        if (!info) return;
                        const card = el.querySelector(`[data-key="${CSS.escape(row.instrument_key)}"]`);
                        if (!card) return;
                        if (info.ltp != null) card.querySelector(".modal-stock-ltp").textContent = "₹" + info.ltp.toFixed(2);
                        if (info.change_pct != null) {
                            const chgEl = card.querySelector(".modal-stock-chg");
                            chgEl.textContent = (info.change_pct >= 0 ? "+" : "") + info.change_pct.toFixed(2) + "%";
                            chgEl.className = "modal-stock-chg " + (info.change_pct >= 0 ? "up" : "down");
                        }
                        if (info.ltp > 0) card.querySelector(".modal-qty").value = Math.max(1, Math.floor(12500 / info.ltp));
                    }).catch(() => {});
            }
        }
        wireModalOrders(el);
    } catch (e) {
        el.innerHTML = '<div class="muted">Failed to load related scripts.</div>';
    }
}

function buildModalStockCard(d, key) {
    const pct = d.change_pct;
    const chgClass = pct != null ? (pct >= 0 ? "up" : "down") : "";
    const chgText  = pct != null ? (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%" : "—";
    const ltp      = d.ltp != null ? d.ltp.toFixed(2) : "—";
    const qty      = d.ltp > 0 ? Math.max(1, Math.floor(12500 / d.ltp)) : 1;
    const sym      = d.trading_symbol || key;
    const name     = encodeURIComponent(d.name || sym);
    const url      = `/static/stock.html?key=${encodeURIComponent(key)}&symbol=${encodeURIComponent(sym)}&name=${name}`;
    return `<div class="modal-stock-card" data-key="${key}" data-url="${url}" style="cursor:pointer;">
        <div class="modal-stock-top">
            <span class="modal-stock-symbol">${sym}</span>
            <span class="modal-stock-name">${d.name || ""}</span>
            <span class="modal-stock-chg ${chgClass}">${chgText}</span>
        </div>
        <div class="modal-stock-ltp">₹${ltp}</div>
        <div class="modal-order-row">
            <input type="number" class="modal-qty" value="${qty}" min="1">
            <button class="btn-long modal-order-btn" data-key="${key}" data-side="long">Long ▲</button>
            <button class="btn-short modal-order-btn" data-key="${key}" data-side="short">Short ▼</button>
            <span class="modal-order-result"></span>
        </div>
    </div>`;
}

function buildModalScannerCard(s) {
    const pct  = s.return_pct;
    const ltp  = s.ltp || s.close_now;
    const qty  = ltp > 0 ? Math.max(1, Math.floor(12500 / ltp)) : 1;
    const chgClass = pct >= 0 ? "up" : "down";
    return `<div class="modal-stock-card">
        <div class="modal-stock-top">
            <span class="modal-stock-symbol">${s.trading_symbol}</span>
            <span class="modal-stock-chg ${chgClass}">${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%</span>
        </div>
        <div class="modal-stock-ltp">₹${ltp != null ? ltp.toFixed(2) : "—"}</div>
        <div class="modal-order-row">
            <input type="number" class="modal-qty" value="${qty}" min="1">
            <button class="btn-long modal-order-btn" data-key="${s.instrument_key}" data-side="long">Long ▲</button>
            <button class="btn-short modal-order-btn" data-key="${s.instrument_key}" data-side="short">Short ▼</button>
            <span class="modal-order-result"></span>
        </div>
    </div>`;
}

function wireModalOrders(container) {
    // Card click → open stock detail page
    container.querySelectorAll(".modal-stock-card[data-url]").forEach(card => {
        card.addEventListener("click", function (e) {
            if (e.target.closest("button") || e.target.closest("input")) return;
            window.open(this.dataset.url, "_blank");
        });
    });

    container.querySelectorAll(".modal-order-btn").forEach(btn => {
        btn.addEventListener("click", async function () {
            const key  = this.dataset.key;
            const side = this.dataset.side;
            const card = this.closest(".modal-stock-card");
            const qty  = parseInt(card.querySelector(".modal-qty").value, 10) || 1;
            const resultEl = card.querySelector(".modal-order-result");
            const allBtns  = card.querySelectorAll(".modal-order-btn");
            allBtns.forEach(b => b.disabled = true);
            resultEl.textContent = "Placing…";
            resultEl.style.color = "var(--text-muted)";
            try {
                const r = await fetch(
                    `/paper/force-entry?instrument_key=${encodeURIComponent(key)}&side=${side}&qty=${qty}`,
                    { method: "POST" }
                );
                const d = await r.json();
                if (d.ok) {
                    resultEl.textContent = `✓ ${side} ${qty} @ ₹${d.entry_price.toFixed(2)}`;
                    resultEl.style.color = "var(--green)";
                } else {
                    resultEl.textContent = d.error || "Failed";
                    resultEl.style.color = "var(--red)";
                }
            } catch (_) {
                resultEl.textContent = "Network error";
                resultEl.style.color = "var(--red)";
            }
            allBtns.forEach(b => b.disabled = false);
        });
    });
}

function closeNewsModal() {
    $("news-modal").classList.add("hidden");
    document.body.style.overflow = "";
}
$("news-modal-close").addEventListener("click", closeNewsModal);
$("news-modal").addEventListener("click", function (e) { if (e.target === this) closeNewsModal(); });
document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeNewsModal(); });
