let dataset = [];
let ahorro = null;
let analitica = null;

let chart = null;
let chart7d = null;
let chartTop5 = null;
let chartHora = null;


// ======================================================
// 1) Cargar predicciones del modelo
// ======================================================
fetch("static/predicciones_multihorizonte.json")
    .then(r => r.json())
    .then(data => {
        dataset = data;
        actualizarDashboard();
    });


// ======================================================
// 2) Cargar estimación de ahorro
// ======================================================
fetch("static/estimacion_ahorro.json")
    .then(r => r.json())
    .then(data => {
        ahorro = data;
        mostrarAhorro();
    });


// ======================================================
// 3) Cargar analítica avanzada
// ======================================================
fetch("static/analitica_consumo.json")
    .then(r => r.json())
    .then(data => {
        analitica = data;
        mostrarAnalitica();
    });



// ===================================================================
// A) MOSTRAR AHORRO
// ===================================================================
function mostrarAhorro() {
    if (!ahorro) return;

    const box = document.getElementById("kpiAhorroBox");
    box.innerHTML = "";

    box.innerHTML += `
        <div class="kpi">
            <div class="kpi-title">Ahorro Real (10%)</div>
            <div class="kpi-value">$${ahorro.ahorro_actual_posible.ahorro_10pct_usd}</div>
        </div>
    `;

    ahorro.ahorro_predicho_futuro.forEach(a => {
        box.innerHTML += `
            <div class="kpi">
                <div class="kpi-title">Ahorro Futuro ${a.horizonte_horas}h</div>
                <div class="kpi-value">$${a.ahorro_10pct_usd}</div>
            </div>
        `;
    });
}



// ===================================================================
// B) MOSTRAR KPIs + GRAFICAS AVANZADAS
// ===================================================================
function mostrarAnalitica() {
    if (!analitica) return;

    // KPIs avanzados
    document.getElementById("kpi_avg_energy").innerText =
        analitica.kpis.promedio_energia_diaria.toFixed(2) + " kWh";

    document.getElementById("kpi_peak_day").innerText =
        analitica.kpis.consumo_maximo_diario.toFixed(2) + " kWh";

    document.getElementById("kpi_expensive_day").innerText =
        analitica.kpis.dia_mas_costoso;


    // GRÁFICAS
    graficarUltimos7Dias();
    graficarTop5();
    graficarPromedioPorHora();
}



// ===================================================================
// C) GRÁFICA — Últimos 7 días
// ===================================================================
function graficarUltimos7Dias() {
    const ctx = document.getElementById("chart_combo");

    if (chart7d) chart7d.destroy();

    chart7d = new Chart(ctx, {
        type: "line",
        data: {
            labels: analitica.ultimos_7_dias.fechas,
            datasets: [
                {
                    label: "Energía (kWh)",
                    data: analitica.ultimos_7_dias.energia,
                    borderColor: "#1976d2",
                    tension: 0.3
                },
                {
                    label: "Agua (L)",
                    data: analitica.ultimos_7_dias.agua,
                    borderColor: "#2e7d32",
                    tension: 0.3
                },
                {
                    label: "Costo (USD)",
                    data: analitica.ultimos_7_dias.costo,
                    borderColor: "#f57c00",
                    tension: 0.3
                }
            ]
        }
    });
}



// ===================================================================
// D) GRÁFICA — Top 5 días costosos
// ===================================================================
function graficarTop5() {
    const ctx = document.getElementById("chart_top5");

    if (chartTop5) chartTop5.destroy();

    chartTop5 = new Chart(ctx, {
        type: "bar",
        data: {
            labels: analitica.top5_costosos.fechas,
            datasets: [
                {
                    label: "Costo (USD)",
                    data: analitica.top5_costosos.costos,
                    backgroundColor: "#d32f2f"
                }
            ]
        }
    });
}



// ===================================================================
// E) GRÁFICA — Promedio por hora del día
// ===================================================================
function graficarPromedioPorHora() {
    const ctx = document.getElementById("chart_hourly");

    if (chartHora) chartHora.destroy();

    chartHora = new Chart(ctx, {
        type: "line",
        data: {
            labels: analitica.promedio_hora.hora,
            datasets: [
                {
                    label: "Energía (kWh)",
                    data: analitica.promedio_hora.energia,
                    borderColor: "#0288d1",
                    tension: 0.2
                },
                {
                    label: "Agua (L)",
                    data: analitica.promedio_hora.agua,
                    borderColor: "#43a047",
                    tension: 0.2
                },
                {
                    label: "Costo (USD)",
                    data: analitica.promedio_hora.costo,
                    borderColor: "#ef6c00",
                    tension: 0.2
                }
            ]
        }
    });
}



// ===================================================================
// F) DASHBOARD ORIGINAL — Predicciones Multihorizonte
// ===================================================================
document.getElementById("horizonSelect").onchange = actualizarDashboard;
document.getElementById("variableSelect").onchange = actualizarDashboard;


function actualizarDashboard() {
    const h = parseInt(document.getElementById("horizonSelect").value);
    const v = document.getElementById("variableSelect").value;

    const datos = dataset.filter(d => d.horizon === h);

    let realKey, predKey, unidad;

    if (v === "energia") { realKey = "energia_real_kwh"; predKey = "energia_pred_kwh"; unidad = "kWh"; }
    else if (v === "agua") { realKey = "agua_real_l"; predKey = "agua_pred_l"; unidad = "Litros"; }
    else { realKey = "costo_real_usd"; predKey = "costo_pred_usd"; unidad = "USD"; }

    // ============================
    // Cálculo métricas
    // ============================
    const errores = datos.map(d => Math.abs(d[realKey] - d[predKey]));
    const mae = errores.reduce((a, b) => a + b, 0) / errores.length;

    const mape = datos.reduce((acc, d) =>
        acc + Math.abs((d[realKey] - d[predKey]) / (Math.abs(d[realKey]) + 1e-8)),
    0) / datos.length * 100;

    document.getElementById("kpi_mae").innerText = mae.toFixed(4) + " " + unidad;
    document.getElementById("kpi_mape").innerText = mape.toFixed(2) + " %";
    document.getElementById("kpi_points").innerText = datos.length;

    // ============================
    // TABLA
    // ============================
    const tbody = document.getElementById("tablaDatos");
    tbody.innerHTML = "";

    datos.slice(0, 200).forEach(d => {
        tbody.innerHTML += `
            <tr>
                <td>${d.timestamp_objetivo}</td>
                <td>${d[realKey].toFixed(3)} ${unidad}</td>
                <td>${d[predKey].toFixed(3)} ${unidad}</td>
                <td>${Math.abs(d[realKey] - d[predKey]).toFixed(3)} ${unidad}</td>
            </tr>`;
    });

    // ============================
    // GRÁFICO PRINCIPAL
    // ============================
    const labels = datos.map(d => d.timestamp_objetivo);
    const reales = datos.map(d => d[realKey]);
    const preds = datos.map(d => d[predKey]);

    if (chart) chart.destroy();

    chart = new Chart(document.getElementById("chart"), {
        type: "line",
        data: {
            labels,
            datasets: [
                { label: "Real", data: reales, borderColor: "#2e7d32", tension: 0.2 },
                { label: "Predicción", data: preds, borderColor: "#1565c0", tension: 0.2 }
            ]
        }
    });
}
