// ==========================================
// FASTAPI URL
// ==========================================

const API_URL =
    window.__API_URL__ ||
    (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
        ? "http://127.0.0.1:8000"
        : window.location.origin);

// ==========================================
// VARIABLES
// ==========================================

let transactionCount = 0;
let history = [];
let fraudChart;

// ==========================================
// SAMPLE DATA
// ==========================================

const normalSample = {
    Time: 50000,
    V1: 0.5,
    V2: 0.2,
    V3: 0.8,
    V4: -0.3,
    V5: 0.4,
    V6: -0.2,
    V7: 0.6,
    V8: 0.1,
    V9: 0.3,
    V10: -0.1,
    Amount: 50
};

const suspiciousSample = {
    Time: 150000,
    V1: -3.2,
    V2: 3.1,
    V3: -2.8,
    V4: 3.0,
    V5: -2.5,
    V6: 1.8,
    V7: -2.9,
    V8: 2.0,
    V9: -2.4,
    V10: 2.8,
    Amount: 1500
};

// ==========================================
// LOAD SAMPLE
// ==========================================

function loadSample() {
    const sample = Math.random() > 0.5 ? normalSample : suspiciousSample;

    for (const key in sample) {
        const input = document.getElementById(key);
        if (input) {
            input.value = sample[key];
        }
    }
}

// ==========================================
// GET FORM DATA
// ==========================================

function getTransactionData() {
    const fields = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "Amount"
    ];

    const data = {};

    fields.forEach(field => {
        data[field] = Number(document.getElementById(field).value);
    });

    return data;
}

// ==========================================
// CHECK FASTAPI
// ==========================================

async function checkBackend() {
    try {
        const response = await fetch(API_URL + "/health");

        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        document.getElementById("apiStatus").textContent = "API Connected";
        document.getElementById("systemStatus").textContent = "Online";
        document.getElementById("apiDot").style.background = "#35d399";
    }
    catch (error) {
        document.getElementById("apiStatus").textContent = "API Offline";
        document.getElementById("systemStatus").textContent = "Offline";
        document.getElementById("apiDot").style.background = "#ff5d6c";
    }
}

// ==========================================
// PREDICTION
// ==========================================

async function predictTransaction() {
    const button = document.getElementById("analyzeButton");
    button.disabled = true;
    button.innerHTML = "Analyzing...";

    const data = getTransactionData();

    try {
        const response = await fetch(API_URL + "/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const result = await response.json();
        displayResult(result);
    }
    catch (error) {
        alert(
            "Cannot connect to FastAPI.\n\n" +
            "Start your backend using:\n" +
            "python -m uvicorn backend.main:app --reload"
        );
        console.error(error);
    }

    button.disabled = false;
    button.innerHTML = 'Analyze Transaction <span>→</span>';
}

// ==========================================
// DISPLAY RESULT
// ==========================================

function displayResult(result) {
    document.getElementById("defaultResult").classList.add("hidden");
    document.getElementById("resultContainer").classList.remove("hidden");

    const probability = Number(result.fraud_probability);
    const isFraud = Number(result.prediction) === 1;

    const resultText = document.getElementById("resultText");
    const riskBadge = document.getElementById("riskBadge");

    if (isFraud) {
        resultText.textContent = "FRAUDULENT";
        resultText.style.color = "#ff5d6c";
    } else {
        resultText.textContent = "LEGITIMATE";
        resultText.style.color = "#35d399";
    }

    if (probability >= 70) {
        riskBadge.textContent = "HIGH RISK";
        riskBadge.style.color = "#ff5d6c";
        riskBadge.style.background = "#29131a";
    } else if (probability >= 35) {
        riskBadge.textContent = "MEDIUM RISK";
        riskBadge.style.color = "#ffb454";
        riskBadge.style.background = "#2a2112";
    } else {
        riskBadge.textContent = "LOW RISK";
        riskBadge.style.color = "#35d399";
        riskBadge.style.background = "#10251f";
    }

    document.getElementById("probability").textContent = probability.toFixed(2) + "%";
    document.getElementById("progressBar").style.width = Math.min(probability, 100) + "%";
    document.getElementById("prediction").textContent = result.prediction;

    transactionCount++;
    document.getElementById("transactionCount").textContent = transactionCount;

    addHistory(isFraud, probability);
    updateChart();
}

// ==========================================
// HISTORY
// ==========================================

function addHistory(isFraud, probability) {
    const item = {
        result: isFraud ? "FRAUDULENT" : "LEGITIMATE",
        probability: probability,
        time: new Date().toLocaleTimeString()
    };

    history.unshift(item);

    if (history.length > 7) {
        history.pop();
    }

    const historyList = document.getElementById("historyList");
    historyList.innerHTML = "";

    history.forEach(item => {
        const row = document.createElement("div");
        row.className = "history-item";

        row.innerHTML = `
            <div>
                <strong>${item.result}</strong>
                <small>${item.time}</small>
            </div>
            <div class="history-right">
                <strong>${item.probability.toFixed(2)}%</strong>
                <small>Fraud probability</small>
            </div>
        `;

        historyList.appendChild(row);
    });
}

// ==========================================
// CLEAR HISTORY
// ==========================================

function clearHistory() {
    history = [];

    document.getElementById("historyList").innerHTML = `
        <div class="empty">
            No predictions yet.
        </div>
    `;

    updateChart();
}

// ==========================================
// RESET FORM
// ==========================================

function resetForm() {
    for (const key in normalSample) {
        const input = document.getElementById(key);
        if (input) {
            input.value = normalSample[key];
        }
    }

    document.getElementById("resultContainer").classList.add("hidden");
    document.getElementById("defaultResult").classList.remove("hidden");

    document.getElementById("probability").textContent = "0%";
    document.getElementById("progressBar").style.width = "0%";
}

// ==========================================
// CHART
// ==========================================

function createChart() {
    const canvas = document.getElementById("fraudChart");

    fraudChart = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels: ["Legitimate", "Fraudulent"],
            datasets: [{
                data: [0, 0],
                backgroundColor: ["#35d399", "#ff5d6c"],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "72%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: getComputedStyle(document.body).getPropertyValue("--muted"),
                        boxWidth: 10,
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

// ==========================================
// UPDATE CHART
// ==========================================

function updateChart() {
    let fraud = 0;
    let legitimate = 0;

    history.forEach(item => {
        if (item.result === "FRAUDULENT") {
            fraud++;
        } else {
            legitimate++;
        }
    });

    fraudChart.data.datasets[0].data = [legitimate, fraud];
    fraudChart.update();
}

// ==========================================
// FORM SUBMIT
// ==========================================

document.getElementById("fraudForm").addEventListener("submit", function (event) {
    event.preventDefault();
    predictTransaction();
});

// ==========================================
// START
// ==========================================

createChart();
checkBackend();

// Check backend every 15 seconds
setInterval(checkBackend, 15000);
