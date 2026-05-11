const authContainer = document.getElementById('auth-container');
const dashboardContainer = document.getElementById('dashboard-container');
const ticksBody = document.getElementById('ticks-body');

let tickState = {};

async function checkAuthStatus() {
    try {
        const response = await fetch('/auth/status');
        const data = await response.json();
        
        if (data.authenticated) {
            authContainer.style.display = 'none';
            dashboardContainer.style.display = 'block';
            startSSE();
        } else {
            authContainer.style.display = 'flex';
            dashboardContainer.style.display = 'none';
        }
    } catch (error) {
        console.error('Auth status check failed:', error);
    }
}

function startSSE() {
    const eventSource = new EventSource('/stream');

    eventSource.onmessage = (event) => {
        const tickData = JSON.parse(event.data);
        updateTicksTable(tickData);
    };

    eventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        eventSource.close();
        // Reconnect after 3 seconds
        setTimeout(startSSE, 3000);
    };
}

function updateTicksTable(data) {
    // Upstox V3 tick data structure in ltpc mode: { "feeds": { "KEY": { "ltpc": { "ltp": 2500.5, ... } } } }
    if (!data.feeds) return;

    for (const [key, feed] of Object.entries(data.feeds)) {
        // Try the flatter structure first (ltpc mode), then fallback to full ff structure
        const ltpData = feed.ltpc || feed.ff?.marketFF?.ltpc;
        if (!ltpData) continue;

        const symbol = key.split('|')[1] || key;
        const ltp = ltpData.ltp;
        const cp = ltpData.cp; // Close price for change calculation
        const changePercent = cp ? (((ltp - cp) / cp) * 100).toFixed(2) : '0.00';
        const time = new Date().toLocaleTimeString();

        let row = document.getElementById(`row-${key.replace(/\|/g, '-')}`);
        let flashClass = '';

        if (tickState[key]) {
            if (ltp > tickState[key].ltp) flashClass = 'flash-up';
            else if (ltp < tickState[key].ltp) flashClass = 'flash-down';
        }

        tickState[key] = { ltp, changePercent };

        const rowHtml = `
            <td>${symbol}</td>
            <td class="${flashClass}">${ltp.toFixed(2)}</td>
            <td class="${parseFloat(changePercent) >= 0 ? 'change-pos' : 'change-neg'}">${changePercent}%</td>
            <td>${time}</td>
        `;

        if (row) {
            row.innerHTML = rowHtml;
            // Remove flash class after animation
            if (flashClass) {
                const ltpCell = row.cells[1];
                setTimeout(() => ltpCell.classList.remove(flashClass), 1000);
            }
        } else {
            row = document.createElement('tr');
            row.id = `row-${key.replace(/\|/g, '-')}`;
            row.innerHTML = rowHtml;
            ticksBody.appendChild(row);
        }
    }
}

// Initial check
checkAuthStatus();
