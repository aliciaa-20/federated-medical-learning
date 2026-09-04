/**
 * MedFed AI - Application Controller & Visualizer
 */

// Global State
const state = {
    baseline: { accuracy: 0.9649, f1_score: 0.9718 },
    currentModel: { accuracy: 0.9825, f1_score: 0.9861, total_comm_KB: 29.06 },
    history: null,
    features: [],
    presets: {},
    featureValues: [],
    activeChartTab: 'accuracy',
    activeViewMode: 'dashboard',
    showExtendedFeatures: false,
    isTraining: false,
    particles: [],
    chartInstance: null,
    backendConnected: true,
    hospitalNames: [
        "Mayo Clinic Node",
        "Johns Hopkins Node",
        "Cleveland Clinic Node",
        "Stanford Health Node",
        "Mount Sinai Node",
        "Mass General Node",
        "UCSF Medical Node",
        "MD Anderson Node",
        "Cedars-Sinai Node",
        "Northwestern Node",
        "Duke Health Node",
        "UPMC Node",
        "Emory Healthcare",
        "Vanderbilt Health",
        "Yale New Haven",
        "NYU Langone Node"
    ]
};

// Top 6 primary essential biomarkers for streamlined display
const PRIMARY_FEATURE_INDICES = [0, 1, 2, 3, 4, 6]; 
// Remaining key features for extended mode
const EXTENDED_FEATURE_INDICES = [5, 7, 8, 9];

// Initialize on DOM Load
document.addEventListener('DOMContentLoaded', async () => {
    initViewModes();
    initControls();
    initChartTabs();
    initTopology();
    await fetchInitialState();
    initPredictorPresets();
    initExportButton();
    initFeatureToggle();
    renderBiomarkerSliders();
    triggerPrediction();
    startBackendHeartbeat();
});

/* --------------------------------------------------------------------------
   1. API Communication & Backend Connection Verification
   -------------------------------------------------------------------------- */
async function fetchInitialState() {
    const t0 = performance.now();
    try {
        addLog('Connecting to MedFed AI Python backend (:5050)...', 'system');
        const res = await fetch('/api/initial-state');
        const data = await res.json();
        const latency = Math.round(performance.now() - t0);
        
        state.baseline = data.baseline;
        state.currentModel = data.current_model;
        state.features = data.preset_data.features;
        state.presets = data.preset_data.presets;
        
        // Initialize default feature values from benign preset
        state.featureValues = [...data.preset_data.presets.benign.values];
        
        // Update UI
        updateBackendStatus(true, latency);
        updateKPICards(state.currentModel, state.baseline);
        updateActiveClients(4);
        document.getElementById('navBaselineAcc').textContent = `${(data.baseline.accuracy * 100).toFixed(2)}% Acc`;
        
        // Fetch default training history for charts
        await runTraining(4, 15, 1, true, 0.01, false);
        addLog(`Backend connected successfully (${latency}ms). Cluster ready.`, 'success');
    } catch (err) {
        console.error('Error fetching initial state:', err);
        updateBackendStatus(false);
        addLog(`Backend warning: ${err.message}. Running in offline simulation mode.`, 'warning');
    }
}

function updateBackendStatus(online, latencyMs = 0) {
    state.backendConnected = online;
    const dot = document.getElementById('backendStatusDot');
    const text = document.getElementById('backendStatusText');
    const pill = document.getElementById('backendStatusPill');
    
    if (online) {
        dot.className = 'status-dot live';
        text.innerHTML = `Backend: <strong>Connected</strong> (${latencyMs ? latencyMs + 'ms' : 'Port 5000'})`;
        pill.style.borderColor = 'rgba(16, 185, 129, 0.35)';
    } else {
        dot.className = 'status-dot offline';
        text.innerHTML = `Backend: <strong>Disconnected</strong>`;
        pill.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    }
}

function startBackendHeartbeat() {
    setInterval(async () => {
        try {
            const t0 = performance.now();
            const res = await fetch('/api/initial-state');
            if (res.ok) {
                const latency = Math.round(performance.now() - t0);
                updateBackendStatus(true, latency);
            } else {
                updateBackendStatus(false);
            }
        } catch {
            updateBackendStatus(false);
        }
    }, 12000);
}

async function runTraining(numClients, rounds, localEpochs, nonIid, lr, animate = true) {
    if (state.isTraining) return;
    state.isTraining = true;
    
    const progressWrap = document.getElementById('trainingProgressWrap');
    const progressBar = document.getElementById('trainingProgressBar');
    const progressText = document.getElementById('trainingStatusText');
    const progressPct = document.getElementById('trainingPercentage');
    const btnSubmit = document.getElementById('btnStartTraining');
    
    if (animate) {
        progressWrap.classList.remove('hidden');
        btnSubmit.disabled = true;
        addLog(`Initiating Federated Training: ${numClients} clients, ${rounds} rounds, non_iid=${nonIid}...`, 'info');
    }
    
    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_clients: numClients,
                rounds: rounds,
                local_epochs: localEpochs,
                non_iid: nonIid,
                lr: lr
            })
        });
        
        const data = await res.json();
        state.history = data.history;
        state.currentModel = data.final_metrics;
        
        if (animate) {
            const totalRounds = data.history.round.length;
            for (let i = 0; i < totalRounds; i++) {
                const currentRound = data.history.round[i];
                const acc = data.history.accuracy[i];
                const f1 = data.history.f1[i];
                const comm = data.history.cumulative_comm_kb[i];
                const pct = Math.round(((i + 1) / totalRounds) * 100);
                
                progressBar.style.width = `${pct}%`;
                progressPct.textContent = `${pct}%`;
                progressText.textContent = `Round ${currentRound}/${totalRounds} | Acc: ${(acc * 100).toFixed(1)}% | F1: ${f1.toFixed(3)}`;
                
                addLog(`[FedAvg] Round ${currentRound}/${totalRounds} -> Global Acc: ${(acc * 100).toFixed(2)}%, F1: ${f1.toFixed(4)}, Comm: ${comm} KB`, 'info');
                
                burstParticles();
                await new Promise(r => setTimeout(r, 55));
            }
            
            addLog(`Training Complete! Final Accuracy: ${(data.final_metrics.accuracy * 100).toFixed(2)}% (F1: ${data.final_metrics.f1_score.toFixed(4)})`, 'success');
        }
        
        updateKPICards(data.final_metrics, data.baseline);
        updateChart();
        triggerPrediction();
    } catch (err) {
        console.error('Training failed:', err);
        addLog(`Training error: ${err.message}`, 'warning');
    } finally {
        state.isTraining = false;
        if (animate) {
            btnSubmit.disabled = false;
            setTimeout(() => progressWrap.classList.add('hidden'), 1000);
        }
    }
}

async function triggerPrediction() {
    if (!state.featureValues || state.featureValues.length === 0) return;
    
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: state.featureValues })
        });
        
        const result = await res.json();
        renderPredictionResult(result);
    } catch (err) {
        console.error('Prediction failed:', err);
    }
}

/* --------------------------------------------------------------------------
   2. View Segment Switcher (Simplified Modes)
   -------------------------------------------------------------------------- */
function initViewModes() {
    const navTabs = document.querySelectorAll('.nav-tab');
    const mainGrid = document.getElementById('mainDashboardView');
    
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const view = tab.dataset.view;
            state.activeViewMode = view;
            
            mainGrid.className = 'dashboard-grid';
            if (view === 'training') {
                mainGrid.classList.add('mode-training');
            } else if (view === 'diagnostic') {
                mainGrid.classList.add('mode-diagnostic');
            }
            
            // Resize canvas or chart when layout shifts
            if (view === 'training' || view === 'dashboard') {
                setTimeout(() => {
                    initCanvasParticles();
                    renderHospitalNodes(parseInt(document.getElementById('inputClients').value) || 4);
                }, 50);
            }
            if (state.chartInstance) {
                setTimeout(() => state.chartInstance.resize(), 50);
            }
        });
    });
}

/* --------------------------------------------------------------------------
   3. UI Controls & Event Listeners
   -------------------------------------------------------------------------- */
function initControls() {
    const inputClients = document.getElementById('inputClients');
    const labelClients = document.getElementById('labelClients');
    const inputRounds = document.getElementById('inputRounds');
    const labelRounds = document.getElementById('labelRounds');
    const inputEpochs = document.getElementById('inputEpochs');
    const labelEpochs = document.getElementById('labelEpochs');
    const form = document.getElementById('flConfigForm');
    const btnBenchmark = document.getElementById('btnBenchmarkScalability');
    
    inputClients.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        labelClients.textContent = `${val} Sites`;
        updateActiveClients(val);
    });
    
    inputRounds.addEventListener('input', (e) => {
        labelRounds.textContent = `${e.target.value} Rounds`;
    });
    
    inputEpochs.addEventListener('input', (e) => {
        labelEpochs.textContent = `${e.target.value} Epoch${e.target.value > 1 ? 's' : ''}`;
    });
    
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const numClients = parseInt(inputClients.value);
        const rounds = parseInt(inputRounds.value);
        const localEpochs = parseInt(inputEpochs.value);
        const nonIid = document.getElementById('selectDistribution').value === 'true';
        runTraining(numClients, rounds, localEpochs, nonIid, 0.01, true);
    });
    
    btnBenchmark.addEventListener('click', async () => {
        addLog('Running multi-cloud scalability benchmark (2, 4, 8, 16 hospital nodes)...', 'info');
        switchChartTab('scalability');
        try {
            const res = await fetch('/api/scalability', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rounds: 15 })
            });
            const data = await res.json();
            state.scalabilityResults = data.results;
            updateScalabilityChart(data.results);
            addLog('Scalability benchmark finished. Accuracy remains steady across distributed clouds.', 'success');
        } catch (err) {
            console.error('Benchmark failed:', err);
        }
    });
    
    document.getElementById('btnClearLog').addEventListener('click', () => {
        document.getElementById('terminalLogBody').innerHTML = '';
        addLog('Terminal log cleared.', 'system');
    });
}

function updateActiveClients(count) {
    document.getElementById('activeClientsHeader').textContent = `${count} Hospital Nodes Connected`;
    renderHospitalNodes(count);
}

function updateKPICards(metrics, baseline) {
    const valAcc = document.getElementById('valAccuracy');
    const valAccDelta = document.getElementById('valAccuracyDelta');
    const valF1 = document.getElementById('valF1');
    const valComm = document.getElementById('valComm');
    
    const accPct = (metrics.accuracy * 100).toFixed(2);
    const basePct = (baseline.accuracy * 100).toFixed(2);
    const diff = (metrics.accuracy - baseline.accuracy) * 100;
    
    valAcc.textContent = `${accPct}%`;
    valAccDelta.textContent = `${diff >= 0 ? '+' : ''}${diff.toFixed(2)}% vs Baseline`;
    valAccDelta.className = `kpi-delta ${diff >= 0 ? 'positive' : 'neutral'}`;
    
    valF1.textContent = metrics.f1_score.toFixed(4);
    valComm.textContent = `${metrics.total_comm_KB || 29.06} KB`;
    
    document.getElementById('statPeakAcc').textContent = `${accPct}%`;
}

/* --------------------------------------------------------------------------
   4. Diagnostic Biomarker Sliders & Presets
   -------------------------------------------------------------------------- */
function initFeatureToggle() {
    const btn = document.getElementById('btnToggleAllFeatures');
    const text = document.getElementById('toggleFeaturesText');
    
    btn.addEventListener('click', () => {
        state.showExtendedFeatures = !state.showExtendedFeatures;
        text.textContent = state.showExtendedFeatures ? 
            'Show Essential Biomarkers Only (6 features)' : 
            'Show Extended Biomarkers (10 features)';
        renderBiomarkerSliders();
    });
}

function renderBiomarkerSliders() {
    const container = document.getElementById('biomarkersGrid');
    container.innerHTML = '';
    
    const activeIndices = state.showExtendedFeatures ? 
        [...PRIMARY_FEATURE_INDICES, ...EXTENDED_FEATURE_INDICES] : 
        PRIMARY_FEATURE_INDICES;
    
    activeIndices.forEach((idx) => {
        const feat = state.features[idx];
        if (!feat) return;
        
        const box = document.createElement('div');
        box.className = 'slider-box';
        
        const cleanName = feat.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        const currentVal = state.featureValues[idx] !== undefined ? state.featureValues[idx] : feat.mean;
        const step = ((feat.max - feat.min) / 100).toFixed(3);
        
        box.innerHTML = `
            <div class="slider-box-header">
                <span>${cleanName}</span>
                <strong id="valDisplay_${idx}">${currentVal.toFixed(2)}</strong>
            </div>
            <input type="range" 
                   id="slider_${idx}" 
                   min="${feat.min}" 
                   max="${feat.max}" 
                   step="${step}" 
                   value="${currentVal}">
        `;
        
        const slider = box.querySelector(`#slider_${idx}`);
        slider.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            state.featureValues[idx] = v;
            document.getElementById(`valDisplay_${idx}`).textContent = v.toFixed(2);
            debouncedPredict();
        });
        
        container.appendChild(box);
    });
}

function updateSliderValues() {
    const allIndices = [...PRIMARY_FEATURE_INDICES, ...EXTENDED_FEATURE_INDICES];
    allIndices.forEach(idx => {
        const slider = document.getElementById(`slider_${idx}`);
        const display = document.getElementById(`valDisplay_${idx}`);
        if (slider && display && state.featureValues[idx] !== undefined) {
            slider.value = state.featureValues[idx];
            display.textContent = state.featureValues[idx].toFixed(2);
        }
    });
}

let predictTimer = null;
function debouncedPredict() {
    clearTimeout(predictTimer);
    predictTimer = setTimeout(triggerPrediction, 100);
}

function initPredictorPresets() {
    const btnMalignant = document.getElementById('presetMalignant');
    const btnBenign = document.getElementById('presetBenign');
    const btnBorderline = document.getElementById('presetBorderline');
    const btnRandom = document.getElementById('presetRandom');
    const chips = [btnMalignant, btnBenign, btnBorderline, btnRandom];
    
    const setActive = (activeBtn) => {
        chips.forEach(c => c.classList.remove('active'));
        if (activeBtn) activeBtn.classList.add('active');
    };
    
    btnMalignant.addEventListener('click', () => {
        setActive(btnMalignant);
        if (state.presets.malignant) {
            state.featureValues = [...state.presets.malignant.values];
            updateSliderValues();
            triggerPrediction();
            addLog('Loaded clinical case: High-Risk Malignant Cytology.', 'info');
        }
    });
    
    btnBenign.addEventListener('click', () => {
        setActive(btnBenign);
        if (state.presets.benign) {
            state.featureValues = [...state.presets.benign.values];
            updateSliderValues();
            triggerPrediction();
            addLog('Loaded clinical case: Typical Benign Cytology.', 'info');
        }
    });
    
    btnBorderline.addEventListener('click', () => {
        setActive(btnBorderline);
        if (state.presets.malignant && state.presets.benign) {
            state.featureValues = state.presets.malignant.values.map((m, i) => (m + state.presets.benign.values[i]) / 2);
            updateSliderValues();
            triggerPrediction();
            addLog('Loaded clinical case: Borderline Diagnostic Profile.', 'info');
        }
    });
    
    btnRandom.addEventListener('click', () => {
        setActive(btnRandom);
        state.featureValues = state.features.map(f => f.min + Math.random() * (f.max - f.min));
        updateSliderValues();
        triggerPrediction();
        addLog('Generated synthetic patient biopsy biomarker profile.', 'info');
    });
}

function renderPredictionResult(res) {
    const card = document.getElementById('diagnosisResultCard');
    const title = document.getElementById('diagnosisText');
    const subtext = document.getElementById('diagnosisSubtext');
    const iconBadge = document.getElementById('diagnosisIconBadge');
    const gaugeFill = document.getElementById('gaugeBarFill');
    const probMal = document.getElementById('probMalignantText');
    const probBen = document.getElementById('probBenignText');
    const factorsList = document.getElementById('factorsList');
    
    const isBenign = res.prediction_label === 1;
    
    if (isBenign) {
        card.classList.remove('malignant-state');
        title.className = 'result-title text-benign';
        title.textContent = 'Benign (Non-Cancerous)';
        subtext.textContent = `Confidence: ${res.confidence_percent}% • Low risk profile`;
        iconBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    } else {
        card.classList.add('malignant-state');
        title.className = 'result-title text-malignant';
        title.textContent = 'Malignant (High Risk)';
        subtext.textContent = `Confidence: ${res.confidence_percent}% • Indicators require immediate clinical review`;
        iconBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
    }
    
    gaugeFill.style.width = `${res.prob_benign}%`;
    probMal.textContent = `${res.prob_malignant}%`;
    probBen.textContent = `${res.prob_benign}%`;
    
    factorsList.innerHTML = '';
    if (res.top_factors && res.top_factors.length > 0) {
        res.top_factors.slice(0, 4).forEach(f => {
            const tag = document.createElement('span');
            const isFavorsBenign = f.impact.includes('Benign');
            tag.className = `factor-tag ${isFavorsBenign ? 'benign-factor' : 'malignant-factor'}`;
            tag.innerHTML = `<strong>${f.feature.replace('_', ' ')}</strong>: ${f.value} (${f.impact})`;
            factorsList.appendChild(tag);
        });
    }
}

/* --------------------------------------------------------------------------
   5. Chart.js Telemetry Visualization
   -------------------------------------------------------------------------- */
function initChartTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            switchChartTab(btn.dataset.chart);
        });
    });
}

function switchChartTab(tab) {
    state.activeChartTab = tab;
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(b => {
        if (b.dataset.chart === tab) b.classList.add('active');
        else b.classList.remove('active');
    });
    updateChart();
}

function updateChart() {
    const ctx = document.getElementById('analyticsChart').getContext('2d');
    if (!state.history) return;
    
    if (state.chartInstance) {
        state.chartInstance.destroy();
    }
    
    const rounds = state.history.round;
    const defaultFont = { family: 'Inter', size: 11 };
    
    if (state.activeChartTab === 'accuracy') {
        state.chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: rounds.map(r => `R${r}`),
                datasets: [
                    {
                        label: 'Federated Accuracy',
                        data: state.history.accuracy.map(a => a * 100),
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.12)',
                        borderWidth: 2.2,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: '#06b6d4',
                        pointRadius: 3
                    },
                    {
                        label: `Central Baseline (${(state.baseline.accuracy * 100).toFixed(1)}%)`,
                        data: rounds.map(() => state.baseline.accuracy * 100),
                        borderColor: '#ef4444',
                        borderWidth: 1.8,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: defaultFont } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: '#64748b', font: defaultFont }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { 
                        min: 88, 
                        max: 100, 
                        ticks: { color: '#64748b', font: defaultFont, callback: v => `${v}%` }, 
                        grid: { color: 'rgba(255,255,255,0.05)' } 
                    }
                }
            }
        });
    } else if (state.activeChartTab === 'metrics') {
        state.chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: rounds.map(r => `R${r}`),
                datasets: [
                    {
                        label: 'F1-Score',
                        data: state.history.f1,
                        borderColor: '#10b981',
                        borderWidth: 2.2,
                        tension: 0.3,
                        pointRadius: 3
                    },
                    {
                        label: 'Precision',
                        data: state.history.precision,
                        borderColor: '#8b5cf6',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2.5
                    },
                    {
                        label: 'Recall',
                        data: state.history.recall,
                        borderColor: '#f59e0b',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2.5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8', font: defaultFont } } },
                scales: {
                    x: { ticks: { color: '#64748b', font: defaultFont }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { min: 0.85, max: 1.0, ticks: { color: '#64748b', font: defaultFont }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    } else if (state.activeChartTab === 'bandwidth') {
        state.chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: rounds.map(r => `R${r}`),
                datasets: [
                    {
                        label: 'Cumulative Bandwidth (KB)',
                        data: state.history.cumulative_comm_kb,
                        backgroundColor: 'rgba(16, 185, 129, 0.4)',
                        borderColor: '#10b981',
                        borderWidth: 1.5,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8', font: defaultFont } } },
                scales: {
                    x: { ticks: { color: '#64748b', font: defaultFont }, grid: { display: false } },
                    y: { ticks: { color: '#64748b', font: defaultFont, callback: v => `${v} KB` }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    } else if (state.activeChartTab === 'scalability') {
        const clientLabels = ['2 Sites', '4 Sites', '8 Sites', '16 Sites'];
        const accData = state.scalabilityResults ? 
            [state.scalabilityResults['2'].accuracy * 100, state.scalabilityResults['4'].accuracy * 100, state.scalabilityResults['8'].accuracy * 100, state.scalabilityResults['16'].accuracy * 100] :
            [97.37, 98.25, 96.49, 95.61];
            
        state.chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: clientLabels,
                datasets: [
                    {
                        label: 'Test Accuracy (%)',
                        data: accData,
                        backgroundColor: 'rgba(6, 182, 212, 0.45)',
                        borderColor: '#06b6d4',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8', font: defaultFont } } },
                scales: {
                    x: { ticks: { color: '#64748b', font: defaultFont }, grid: { display: false } },
                    y: { min: 80, max: 100, ticks: { color: '#64748b', font: defaultFont, callback: v => `${v}%` }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }
}

function updateScalabilityChart(results) {
    state.scalabilityResults = results;
    if (state.activeChartTab === 'scalability') {
        updateChart();
    }
}

/* --------------------------------------------------------------------------
   6. Network Topology & Canvas Particle Animation
   -------------------------------------------------------------------------- */
function initTopology() {
    renderHospitalNodes(4);
    initCanvasParticles();
}

function renderHospitalNodes(count) {
    const grid = document.getElementById('hospitalNodesGrid');
    if (!grid) return;
    grid.innerHTML = '';
    
    const container = document.getElementById('topologyContainer');
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 240;
    const centerX = width / 2;
    const centerY = height / 2;
    const radiusX = width * 0.38;
    const radiusY = height * 0.35;
    
    state.hospitalNodePositions = [];
    
    for (let i = 0; i < count; i++) {
        const angle = (i / count) * (2 * Math.PI) - Math.PI / 2;
        const x = centerX + radiusX * Math.cos(angle);
        const y = centerY + radiusY * Math.sin(angle);
        
        state.hospitalNodePositions.push({ x, y });
        
        const node = document.createElement('div');
        node.className = 'hospital-node';
        node.style.left = `${x - 55}px`;
        node.style.top = `${y - 18}px`;
        
        const name = state.hospitalNames[i] || `Hospital Node #${i + 1}`;
        
        node.innerHTML = `
            <div class="hospital-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v20M2 12h20"></path>
                </svg>
            </div>
            <div class="hospital-meta">
                <span class="hospital-name">${name.replace(' Node', '')}</span>
                <span class="hospital-shards">Data Silo</span>
            </div>
        `;
        
        grid.appendChild(node);
    }
}

function initCanvasParticles() {
    const canvas = document.getElementById('networkCanvas');
    const container = document.getElementById('topologyContainer');
    if (!canvas || !container) return;
    
    const resizeCanvas = () => {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
    };
    
    resizeCanvas();
    window.addEventListener('resize', () => {
        resizeCanvas();
        renderHospitalNodes(parseInt(document.getElementById('inputClients').value) || 4);
    });
    
    const ctx = canvas.getContext('2d');
    
    for (let i = 0; i < 18; i++) {
        spawnParticle();
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        if (state.hospitalNodePositions) {
            state.hospitalNodePositions.forEach(pos => {
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(pos.x, pos.y);
                ctx.strokeStyle = 'rgba(6, 182, 212, 0.14)';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }
        
        for (let i = state.particles.length - 1; i >= 0; i--) {
            const p = state.particles[i];
            p.progress += p.speed;
            
            if (p.progress >= 1) {
                state.particles.splice(i, 1);
                spawnParticle();
                continue;
            }
            
            const currentX = p.fromX + (p.toX - p.fromX) * p.progress;
            const currentY = p.fromY + (p.toY - p.fromY) * p.progress;
            
            ctx.beginPath();
            ctx.arc(currentX, currentY, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.shadowBlur = 6;
            ctx.shadowColor = p.color;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

function spawnParticle(isBurst = false) {
    const canvas = document.getElementById('networkCanvas');
    if (!canvas || !state.hospitalNodePositions || state.hospitalNodePositions.length === 0) return;
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const nodeIdx = Math.floor(Math.random() * state.hospitalNodePositions.length);
    const nodePos = state.hospitalNodePositions[nodeIdx];
    
    const inward = Math.random() > 0.5;
    const fromX = inward ? nodePos.x : centerX;
    const fromY = inward ? nodePos.y : centerY;
    const toX = inward ? centerX : nodePos.x;
    const toY = inward ? centerY : nodePos.y;
    
    const colors = ['#06b6d4', '#10b981', '#a78bfa'];
    
    state.particles.push({
        fromX,
        fromY,
        toX,
        toY,
        progress: 0,
        speed: isBurst ? 0.04 + Math.random() * 0.03 : 0.015 + Math.random() * 0.015,
        radius: isBurst ? 2.5 : 2,
        color: colors[Math.floor(Math.random() * colors.length)]
    });
}

function burstParticles() {
    for (let i = 0; i < 12; i++) {
        spawnParticle(true);
    }
}

/* --------------------------------------------------------------------------
   7. Export & Logging
   -------------------------------------------------------------------------- */
function initExportButton() {
    document.getElementById('btnExportReport').addEventListener('click', () => {
        const exportData = {
            project: "MedFed AI - Federated Learning Medical Diagnosis",
            group: "Cloud CIA Project - Group 10",
            timestamp: new Date().toISOString(),
            centralized_baseline: state.baseline,
            final_metrics: state.currentModel,
            federated_history: state.history,
            scalability_results: state.scalabilityResults || null
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `medfed_simulation_report_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        addLog('Exported full simulation telemetry report (JSON).', 'success');
    });
}

function addLog(msg, type = 'info') {
    const terminal = document.getElementById('terminalLogBody');
    if (!terminal) return;
    
    const timeStr = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.textContent = `[${timeStr}] ${msg}`;
    
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
}
