const statusDot = document.querySelector('.dot');
const statusText = document.querySelector('.status-text');
const authOverlay = document.getElementById('auth-overlay');
const logContainer = document.getElementById('log-container');

function addLog(message) {
    const entry = document.createElement('p');
    entry.className = 'log-entry';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.prepend(entry);
}

function updateTicker(symbol, data) {
    const card = document.getElementById(`card-${symbol}`);
    if (!card) return;

    const priceEl = card.querySelector('.ticker-price');
    const changeEl = card.querySelector('.ticker-change');

    const lp = data.lp || 0;
    const cp = data.cp || 0;
    const change = cp ? ((lp - cp) / cp * 100).toFixed(2) : '0.00';

    const prevPrice = parseFloat(priceEl.textContent) || 0;

    // Update classes for color coding
    card.classList.remove('price-up', 'price-down');
    if (lp > prevPrice) card.classList.add('price-up');
    else if (lp < prevPrice) card.classList.add('price-down');

    changeEl.classList.remove('change-up', 'change-down');
    if (change >= 0) changeEl.classList.add('change-up');
    else changeEl.classList.add('change-down');

    // Set text
    priceEl.textContent = lp.toFixed(2);
    changeEl.textContent = `${change > 0 ? '+' : ''}${change}%`;
}

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

    socket.onopen = () => {
        document.getElementById('status').classList.add('connected');
        statusText.textContent = 'Live Feed Connected';
        addLog('Connected to backend WebSocket.');
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error === 'unauthorized') {
            authOverlay.classList.remove('hidden');
            addLog('Unauthorized: Authentication required.');
            return;
        }

        for (const [symbol, values] of Object.entries(data)) {
            updateTicker(symbol, values);
        }
    };

    socket.onclose = () => {
        document.getElementById('status').classList.remove('connected');
        statusText.textContent = 'Disconnected. Retrying...';
        addLog('WebSocket connection closed. Reconnecting in 5s...');
        setTimeout(connect, 5000);
    };

    socket.onerror = (error) => {
        addLog('WebSocket error occurred.');
        console.error('WebSocket Error:', error);
    };
}

// Initial connection
connect();
