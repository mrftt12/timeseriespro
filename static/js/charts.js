// Chart utilities for TimeSeries Forecasting Platform

const ChartUtils = {
    // Default chart configuration
    defaultConfig: {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'forecast_chart',
            height: 500,
            width: 1000,
            scale: 1
        }
    },

    // Default layout configuration
    defaultLayout: {
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Inter, sans-serif',
            size: 12,
            color: '#374151'
        },
        xaxis: {
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db',
            tickcolor: '#d1d5db',
            titlefont: { size: 14, color: '#1f2937' }
        },
        yaxis: {
            gridcolor: '#e5e7eb',
            linecolor: '#d1d5db',
            tickcolor: '#d1d5db',
            titlefont: { size: 14, color: '#1f2937' }
        },
        legend: {
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: '#d1d5db',
            borderwidth: 1
        },
        hovermode: 'x unified'
    },

    // Color palette
    colors: {
        primary: '#3b82f6',
        teal: '#14b8a6',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        secondary: '#6b7280'
    }
};

// Create forecast visualization chart
function createForecastChart(data, containerId) {
    const traces = [];

    // Historical data trace
    if (data.historical && data.historical.dates && data.historical.values) {
        traces.push({
            x: data.historical.dates,
            y: data.historical.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Historical Data',
            line: {
                color: ChartUtils.colors.primary,
                width: 2
            },
            hovertemplate: '<b>Historical</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
        });
    }

    // Test data - actual vs predicted
    if (data.test && data.test.dates) {
        if (data.test.actual) {
            traces.push({
                x: data.test.dates,
                y: data.test.actual,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual (Test)',
                line: {
                    color: ChartUtils.colors.success,
                    width: 2
                },
                marker: {
                    size: 4,
                    color: ChartUtils.colors.success
                },
                hovertemplate: '<b>Actual (Test)</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
            });
        }

        if (data.test.predicted) {
            traces.push({
                x: data.test.dates,
                y: data.test.predicted,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted (Test)',
                line: {
                    color: ChartUtils.colors.warning,
                    width: 2,
                    dash: 'dot'
                },
                marker: {
                    size: 4,
                    color: ChartUtils.colors.warning
                },
                hovertemplate: '<b>Predicted (Test)</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
            });
        }
    }

    // Future forecast
    if (data.forecast && data.forecast.dates && data.forecast.values) {
        // Confidence interval
        if (data.forecast.confidence_lower && data.forecast.confidence_upper) {
            traces.push({
                x: data.forecast.dates.concat(data.forecast.dates.slice().reverse()),
                y: data.forecast.confidence_upper.concat(data.forecast.confidence_lower.slice().reverse()),
                type: 'scatter',
                fill: 'toself',
                fillcolor: 'rgba(20, 184, 166, 0.2)',
                line: { color: 'rgba(20, 184, 166, 0)' },
                name: 'Confidence Interval',
                showlegend: true,
                hoverinfo: 'skip'
            });
        }

        // Forecast line
        traces.push({
            x: data.forecast.dates,
            y: data.forecast.values,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Forecast',
            line: {
                color: ChartUtils.colors.teal,
                width: 3
            },
            marker: {
                size: 5,
                color: ChartUtils.colors.teal
            },
            hovertemplate: '<b>Forecast</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
        });
    }

    const layout = {
        ...ChartUtils.defaultLayout,
        title: {
            text: 'Time Series Forecast',
            font: { size: 16, color: '#1f2937' }
        },
        xaxis: {
            ...ChartUtils.defaultLayout.xaxis,
            title: 'Date',
            type: 'date'
        },
        yaxis: {
            ...ChartUtils.defaultLayout.yaxis,
            title: 'Value'
        },
        showlegend: true,
        margin: { l: 60, r: 30, t: 60, b: 60 }
    };

    Plotly.newPlot(containerId, traces, layout, ChartUtils.defaultConfig);
}

// Create metrics comparison chart
function createMetricsComparisonChart(models, containerId) {
    if (!models || models.length === 0) return;

    const modelNames = models.map(m => m.model_name);
    const rmseValues = models.map(m => m.rmse || 0);
    const maeValues = models.map(m => m.mae || 0);
    const mapeValues = models.map(m => m.mape || 0);

    const traces = [
        {
            x: modelNames,
            y: rmseValues,
            name: 'RMSE',
            type: 'bar',
            marker: {
                color: ChartUtils.colors.primary,
                opacity: 0.8
            },
            hovertemplate: '<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>'
        },
        {
            x: modelNames,
            y: maeValues,
            name: 'MAE',
            type: 'bar',
            marker: {
                color: ChartUtils.colors.teal,
                opacity: 0.8
            },
            hovertemplate: '<b>%{x}</b><br>MAE: %{y:.4f}<extra></extra>'
        },
        {
            x: modelNames,
            y: mapeValues,
            name: 'MAPE (%)',
            type: 'bar',
            marker: {
                color: ChartUtils.colors.warning,
                opacity: 0.8
            },
            hovertemplate: '<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>'
        }
    ];

    const layout = {
        ...ChartUtils.defaultLayout,
        title: {
            text: 'Model Performance Comparison',
            font: { size: 16, color: '#1f2937' }
        },
        xaxis: {
            ...ChartUtils.defaultLayout.xaxis,
            title: 'Models'
        },
        yaxis: {
            ...ChartUtils.defaultLayout.yaxis,
            title: 'Metric Value'
        },
        barmode: 'group',
        showlegend: true,
        margin: { l: 60, r: 30, t: 60, b: 100 }
    };

    Plotly.newPlot(containerId, traces, layout, ChartUtils.defaultConfig);
}

// Create time series decomposition chart
function createDecompositionChart(data, containerId) {
    const traces = [];
    
    if (data.original) {
        traces.push({
            x: data.dates,
            y: data.original,
            type: 'scatter',
            mode: 'lines',
            name: 'Original',
            line: { color: ChartUtils.colors.primary, width: 1 },
            yaxis: 'y1'
        });
    }

    if (data.trend) {
        traces.push({
            x: data.dates,
            y: data.trend,
            type: 'scatter',
            mode: 'lines',
            name: 'Trend',
            line: { color: ChartUtils.colors.success, width: 2 },
            yaxis: 'y2'
        });
    }

    if (data.seasonal) {
        traces.push({
            x: data.dates,
            y: data.seasonal,
            type: 'scatter',
            mode: 'lines',
            name: 'Seasonal',
            line: { color: ChartUtils.colors.teal, width: 1 },
            yaxis: 'y3'
        });
    }

    if (data.residual) {
        traces.push({
            x: data.dates,
            y: data.residual,
            type: 'scatter',
            mode: 'lines',
            name: 'Residual',
            line: { color: ChartUtils.colors.secondary, width: 1 },
            yaxis: 'y4'
        });
    }

    const layout = {
        ...ChartUtils.defaultLayout,
        title: {
            text: 'Time Series Decomposition',
            font: { size: 16, color: '#1f2937' }
        },
        xaxis: {
            ...ChartUtils.defaultLayout.xaxis,
            title: 'Date',
            type: 'date',
            domain: [0, 1]
        },
        yaxis: { 
            domain: [0.75, 1], 
            title: 'Original',
            titlefont: { color: ChartUtils.colors.primary }
        },
        yaxis2: { 
            domain: [0.5, 0.7], 
            title: 'Trend',
            titlefont: { color: ChartUtils.colors.success }
        },
        yaxis3: { 
            domain: [0.25, 0.45], 
            title: 'Seasonal',
            titlefont: { color: ChartUtils.colors.teal }
        },
        yaxis4: { 
            domain: [0, 0.2], 
            title: 'Residual',
            titlefont: { color: ChartUtils.colors.secondary }
        },
        showlegend: true,
        height: 600,
        margin: { l: 80, r: 30, t: 60, b: 60 }
    };

    Plotly.newPlot(containerId, traces, layout, ChartUtils.defaultConfig);
}

// Create residual analysis chart
function createResidualChart(data, containerId) {
    const traces = [
        {
            x: data.predicted,
            y: data.residuals,
            type: 'scatter',
            mode: 'markers',
            name: 'Residuals',
            marker: {
                color: ChartUtils.colors.primary,
                size: 6,
                opacity: 0.7
            },
            hovertemplate: 'Predicted: %{x:.4f}<br>Residual: %{y:.4f}<extra></extra>'
        },
        {
            x: [Math.min(...data.predicted), Math.max(...data.predicted)],
            y: [0, 0],
            type: 'scatter',
            mode: 'lines',
            name: 'Zero Line',
            line: {
                color: ChartUtils.colors.danger,
                width: 2,
                dash: 'dash'
            },
            showlegend: false
        }
    ];

    const layout = {
        ...ChartUtils.defaultLayout,
        title: {
            text: 'Residual Analysis',
            font: { size: 16, color: '#1f2937' }
        },
        xaxis: {
            ...ChartUtils.defaultLayout.xaxis,
            title: 'Predicted Values'
        },
        yaxis: {
            ...ChartUtils.defaultLayout.yaxis,
            title: 'Residuals'
        },
        showlegend: false,
        margin: { l: 60, r: 30, t: 60, b: 60 }
    };

    Plotly.newPlot(containerId, traces, layout, ChartUtils.defaultConfig);
}

// Create model performance radar chart
function createPerformanceRadarChart(models, containerId) {
    if (!models || models.length === 0) return;

    const traces = models.map((model, index) => {
        // Normalize metrics to 0-1 scale for radar chart
        const maxRMSE = Math.max(...models.map(m => m.rmse || 0));
        const maxMAE = Math.max(...models.map(m => m.mae || 0));
        const maxMAPE = Math.max(...models.map(m => m.mape || 0));

        return {
            type: 'scatterpolar',
            r: [
                1 - (model.rmse || 0) / maxRMSE, // Invert so higher is better
                1 - (model.mae || 0) / maxMAE,   // Invert so higher is better
                1 - (model.mape || 0) / maxMAPE, // Invert so higher is better
                model.r2_score || 0
            ],
            theta: ['RMSE (inv)', 'MAE (inv)', 'MAPE (inv)', 'R² Score'],
            fill: 'toself',
            name: model.model_name,
            line: { color: Object.values(ChartUtils.colors)[index % Object.keys(ChartUtils.colors).length] }
        };
    });

    const layout = {
        ...ChartUtils.defaultLayout,
        title: {
            text: 'Model Performance Comparison (Radar)',
            font: { size: 16, color: '#1f2937' }
        },
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 1]
            }
        },
        showlegend: true,
        margin: { l: 60, r: 60, t: 60, b: 60 }
    };

    Plotly.newPlot(containerId, traces, layout, ChartUtils.defaultConfig);
}

// Export chart utilities
window.ChartUtils = ChartUtils;
window.createForecastChart = createForecastChart;
window.createMetricsComparisonChart = createMetricsComparisonChart;
window.createDecompositionChart = createDecompositionChart;
window.createResidualChart = createResidualChart;
window.createPerformanceRadarChart = createPerformanceRadarChart;
