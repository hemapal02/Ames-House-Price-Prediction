from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variable to store data
data = None

def load_and_process_data():
    """Load and process the house price data"""
    global data
    try:
        # Load data - you'll need to update this path
        data = pd.read_csv('data.csv')  # Update with your actual file path
        
        # Basic data processing
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create sample data for demonstration
        np.random.seed(42)
        sample_size = 1460
        
        data = pd.DataFrame({
            'SalePrice': np.random.normal(180000, 50000, sample_size),
            'OverallQual': np.random.randint(1, 11, sample_size),
            'GrLivArea': np.random.normal(1500, 500, sample_size),
            'GarageArea': np.random.normal(500, 200, sample_size),
            'GarageCars': np.random.randint(0, 4, sample_size),
            'TotalBsmtSF': np.random.normal(1000, 300, sample_size),
            '1stFlrSF': np.random.normal(1000, 300, sample_size),
            'FullBath': np.random.randint(1, 4, sample_size),
            'YearBuilt': np.random.randint(1900, 2020, sample_size),
            'TotRmsAbvGrd': np.random.randint(4, 12, sample_size),
            'WoodDeckSF': np.random.normal(100, 150, sample_size),
            'LotFrontage': np.random.normal(70, 20, sample_size),
            'BsmtFinSF1': np.random.normal(400, 200, sample_size),
            '2ndFlrSF': np.random.normal(300, 400, sample_size),
            'OpenPorchSF': np.random.normal(50, 100, sample_size),
            'HalfBath': np.random.randint(0, 3, sample_size),
            'LotArea': np.random.normal(10000, 5000, sample_size),
            'BsmtFullBath': np.random.randint(0, 3, sample_size),
            'BsmtUnfSF': np.random.normal(500, 300, sample_size),
            'BedroomAbvGr': np.random.randint(1, 6, sample_size),
            'ScreenPorch': np.random.normal(20, 50, sample_size),
            'PoolArea': np.random.normal(10, 100, sample_size),
            'MoSold': np.random.randint(1, 13, sample_size),
            '3SsnPorch': np.random.normal(5, 20, sample_size),
            'BsmtHalfBath': np.random.randint(0, 2, sample_size),
            'MiscVal': np.random.normal(50, 200, sample_size),
            'Id': range(1, sample_size + 1),
            'LowQualFinSF': np.random.normal(10, 50, sample_size),
            'YrSold': np.random.randint(2006, 2021, sample_size),
            'OverallCond': np.random.randint(1, 11, sample_size),
            'MSSubClass': np.random.randint(20, 200, sample_size),
            'EnclosedPorch': np.random.normal(20, 100, sample_size),
            'KitchenAbvGr': np.random.randint(1, 3, sample_size),
        })
        
        # Add some categorical columns with missing values
        data['FireplaceQu'] = np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], 
                                              sample_size, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        data['Fence'] = np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', np.nan], 
                                        sample_size, p=[0.05, 0.1, 0.05, 0.1, 0.7])
        data['Alley'] = np.random.choice(['Grvl', 'Pave', np.nan], 
                                        sample_size, p=[0.05, 0.05, 0.9])
        data['MiscFeature'] = np.random.choice(['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', np.nan], 
                                              sample_size, p=[0.01, 0.02, 0.02, 0.03, 0.01, 0.91])
        data['PoolQC'] = np.random.choice(['Ex', 'Fa', 'Gd', np.nan], 
                                         sample_size, p=[0.02, 0.02, 0.02, 0.94])
        
        # Make sure SalePrice is positive
        data['SalePrice'] = np.abs(data['SalePrice'])
        data['GrLivArea'] = np.abs(data['GrLivArea'])
        data['GarageArea'] = np.abs(data['GarageArea'])
        data['TotalBsmtSF'] = np.abs(data['TotalBsmtSF'])
        
        return True

def get_data_summary():
    """Get basic data summary statistics"""
    if data is None:
        return {}
    
    return {
        'total_houses': len(data),
        'avg_price': f"${data['SalePrice'].mean():,.0f}",
        'median_price': f"${data['SalePrice'].median():,.0f}",
        'price_std': f"${data['SalePrice'].std():,.0f}",
        'min_price': f"${data['SalePrice'].min():,.0f}",
        'max_price': f"${data['SalePrice'].max():,.0f}",
        'columns': list(data.columns),
        'shape': data.shape
    }

def get_missing_data():
    """Get missing data information"""
    if data is None:
        return []
    
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    missing_data = []
    
    for col, count in missing.items():
        percentage = (count / len(data)) * 100
        missing_data.append({
            'feature': col,
            'missing': int(count),
            'percentage': round(percentage, 1)
        })
    
    return sorted(missing_data, key=lambda x: x['missing'], reverse=True)

def get_correlation_data():
    """Get correlation data for numeric features"""
    if data is None:
        return {}
    
    numeric_features = data.select_dtypes(include=[np.number])
    correlation = numeric_features.corr()
    
    # Get top correlations with SalePrice
    sale_price_corr = correlation['SalePrice'].sort_values(ascending=False)
    
    # Get top features (excluding SalePrice itself)
    top_features = []
    for feature, corr_value in sale_price_corr.items():
        if feature != 'SalePrice' and not pd.isna(corr_value):
            top_features.append({
                'name': feature,
                'correlation': round(corr_value, 3)
            })
    
    return {
        'top_features': top_features[:10],
        'correlation_matrix': correlation.round(3).to_dict()
    }

def remove_outliers():
    """Remove outliers using IQR method"""
    global data
    if data is None:
        return {}
    
    original_count = len(data)
    
    # Calculate quartiles and IQR
    first_quartile = data['SalePrice'].quantile(0.25)
    third_quartile = data['SalePrice'].quantile(0.75)
    IQR = third_quartile - first_quartile
    
    # Define boundary (using 3*IQR as in original code)
    new_boundary = third_quartile + 3 * IQR
    
    # Remove outliers
    outliers_mask = data['SalePrice'] > new_boundary
    outliers_count = outliers_mask.sum()
    
    data = data[~outliers_mask].copy()
    
    return {
        'original_count': original_count,
        'outliers_removed': int(outliers_count),
        'final_count': len(data),
        'boundary': f"${new_boundary:,.0f}",
        'q1': f"${first_quartile:,.0f}",
        'q3': f"${third_quartile:,.0f}",
        'iqr': f"${IQR:,.0f}"
    }

def remove_features():
    """Remove features as per original analysis"""
    global data
    if data is None:
        return {}
    
    # Features to remove as per original code
    cols_to_remove = [
        'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 
        'HalfBath', 'LotArea', 'BsmtFullBath', 'BsmtUnfSF', 'BedroomAbvGr', 
        'ScreenPorch', 'PoolArea', 'MoSold', '3SsnPorch', 'BsmtHalfBath', 
        'MiscVal', 'Id', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass', 
        'EnclosedPorch', 'KitchenAbvGr', 'FireplaceQu', 'Fence', 'Alley', 
        'MiscFeature', 'PoolQC', 'GarageCars', '1stFlrSF', 'FullBath'
    ]
    
    # Only remove columns that exist in the dataset
    columns_to_drop_existing = [col for col in cols_to_remove if col in data.columns]
    
    original_columns = len(data.columns)
    data.drop(columns_to_drop_existing, axis=1, inplace=True)
    
    return {
        'original_features': original_columns,
        'removed_features': len(columns_to_drop_existing),
        'final_features': len(data.columns),
        'removed_list': columns_to_drop_existing,
        'remaining_features': list(data.columns)
    }

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data-summary')
def api_data_summary():
    """API endpoint for data summary"""
    return jsonify(get_data_summary())

@app.route('/api/missing-data')
def api_missing_data():
    """API endpoint for missing data"""
    return jsonify(get_missing_data())

@app.route('/api/correlation')
def api_correlation():
    """API endpoint for correlation data"""
    return jsonify(get_correlation_data())

@app.route('/api/price-distribution')
def api_price_distribution():
    """API endpoint for price distribution data"""
    if data is None:
        return jsonify({'error': 'No data available'})
    
    prices = data['SalePrice'].values
    
    # Create histogram data
    hist, bin_edges = np.histogram(prices, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return jsonify({
        'prices': prices.tolist(),
        'histogram': {
            'counts': hist.tolist(),
            'bins': bin_centers.tolist(),
            'bin_edges': bin_edges.tolist()
        },
        'stats': {
            'mean': float(prices.mean()),
            'median': float(np.median(prices)),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max())
        }
    })

@app.route('/api/scatter-data')
def api_scatter_data():
    """API endpoint for scatter plot data"""
    if data is None:
        return jsonify({'error': 'No data available'})
    
    feature = request.args.get('feature', 'OverallQual')
    
    if feature not in data.columns:
        return jsonify({'error': f'Feature {feature} not found'})
    
    x_data = data[feature].values
    y_data = data['SalePrice'].values
    
    # Remove any NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[mask]
    y_data = y_data[mask]
    
    return jsonify({
        'x_data': x_data.tolist(),
        'y_data': y_data.tolist(),
        'feature_name': feature
    })

@app.route('/api/outlier-analysis')
def api_outlier_analysis():
    """API endpoint for outlier analysis"""
    return jsonify(remove_outliers())

@app.route('/api/feature-engineering')
def api_feature_engineering():
    """API endpoint for feature engineering"""
    return jsonify(remove_features())

@app.route('/api/box-plot-data')
def api_box_plot_data():
    """API endpoint for box plot data"""
    if data is None:
        return jsonify({'error': 'No data available'})
    
    prices = data['SalePrice'].values
    
    # Calculate quartiles
    q1 = np.percentile(prices, 25)
    q2 = np.percentile(prices, 50)  # median
    q3 = np.percentile(prices, 75)
    iqr = q3 - q1
    
    # Calculate outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Find outliers
    outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
    
    return jsonify({
        'q1': float(q1),
        'q2': float(q2),
        'q3': float(q3),
        'iqr': float(iqr),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outliers': outliers.tolist(),
        'whiskers': {
            'lower': float(np.min(prices[prices >= lower_bound])),
            'upper': float(np.max(prices[prices <= upper_bound]))
        }
    })

@app.route('/api/process-all')
def api_process_all():
    """API endpoint to run the complete analysis pipeline"""
    try:
        # Step 1: Load data
        load_success = load_and_process_data()
        if not load_success:
            return jsonify({'error': 'Failed to load data'})
        
        # Step 2: Get initial summary
        initial_summary = get_data_summary()
        
        # Step 3: Analyze missing data
        missing_data = get_missing_data()
        
        # Step 4: Get correlation analysis
        correlation_data = get_correlation_data()
        
        # Step 5: Remove outliers
        outlier_info = remove_outliers()
        
        # Step 6: Feature engineering
        feature_info = remove_features()
        
        # Step 7: Final summary
        final_summary = get_data_summary()
        
        return jsonify({
            'success': True,
            'initial_summary': initial_summary,
            'missing_data': missing_data,
            'correlation_data': correlation_data,
            'outlier_info': outlier_info,
            'feature_info': feature_info,
            'final_summary': final_summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Create the HTML template
template_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Analysis Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }

        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }

        .card h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5rem;
            text-align: center;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: scale(1.05);
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }

        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        select, input, button {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }

        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 600;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }

        .error {
            color: #dc3545;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }

        .success {
            color: #155724;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .data-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .info-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }

        .info-label {
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
        }

        .info-value {
            font-size: 1.2rem;
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 House Price Analysis Dashboard</h1>
            <p>Python Flask Backend with Interactive Analysis</p>
        </div>

        <div class="controls">
            <button onclick="runCompleteAnalysis()" id="runAnalysisBtn">🚀 Run Complete Analysis</button>
            <button onclick="loadData()" id="loadDataBtn">📊 Load Data Summary</button>
        </div>

        <div id="statusMessage"></div>

        <div class="stats-grid" id="statsGrid" style="display: none;">
            <div class="stat-card">
                <div class="stat-value" id="totalHouses">-</div>
                <div class="stat-label">Total Houses</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgPrice">-</div>
                <div class="stat-label">Average Price</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="medianPrice">-</div>
                <div class="stat-label">Median Price</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="priceStd">-</div>
                <div class="stat-label">Price Std Dev</div>
            </div>
        </div>

        <div class="dashboard-grid">
            <!-- Price Distribution -->
            <div class="card">
                <h3>📊 Sale Price Distribution</h3>
                <div class="chart-container">
                    <canvas id="priceDistribution"></canvas>
                </div>
            </div>

            <!-- Feature Correlation -->
            <div class="card">
                <h3>🎯 Top Correlated Features</h3>
                <div id="correlationList" class="loading">Click "Run Complete Analysis" to load data</div>
            </div>

            <!-- Scatter Plot -->
            <div class="card">
                <h3>📈 Feature vs Price Analysis</h3>
                <div class="controls">
                    <div class="control-group">
                        <label>Feature:</label>
                        <select id="featureSelect">
                            <option value="OverallQual">Overall Quality</option>
                            <option value="GrLivArea">Living Area</option>
                            <option value="GarageArea">Garage Area</option>
                            <option value="YearBuilt">Year Built</option>
                        </select>
                    </div>
                    <button onclick="updateScatterPlot()">Update Plot</button>
                </div>
                <div class="chart-container">
                    <canvas id="scatterPlot"></canvas>
                </div>
            </div>

            <!-- Missing Data -->
            <div class="card">
                <h3>❌ Missing Data Analysis</h3>
                <div id="missingDataList" class="loading">Click "Run Complete Analysis" to load data</div>
            </div>

            <!-- Box Plot -->
            <div class="card">
                <h3>📦 Price Distribution Box Plot</h3>
                <div class="chart-container">
                    <canvas id="boxPlot"></canvas>
                </div>
            </div>

            <!-- Outlier Analysis -->
            <div class="card">
                <h3>🎯 Outlier Analysis</h3>
                <div id="outlierAnalysis" class="loading">Click "Run Complete Analysis" to load data</div>
            </div>

            <!-- Feature Engineering -->
            <div class="card full-width">
                <h3>🧹 Feature Engineering Summary</h3>
                <div id="featureEngineering" class="loading">Click "Run Complete Analysis" to load data</div>
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        let charts = {};

        async function showStatus(message, type = 'info') {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.innerHTML = `<div class="${type === 'error' ? 'error' : 'success'}">${message}</div>`;
            if (type !== 'error') {
                setTimeout(() => statusDiv.innerHTML = '', 5000);
            }
        }

        async function loadData() {
            try {
                showStatus('Loading data summary...', 'info');
                const response = await fetch('/api/data-summary');
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }

                // Update stats
                document.getElementById('totalHouses').textContent = data.total_houses;
                document.getElementById('avgPrice').textContent = data.avg_price;
                document.getElementById('medianPrice').textContent = data.median_price;
                document.getElementById('priceStd').textContent = data.price_std;
                
                document.getElementById('statsGrid').style.display = 'grid';
                showStatus('Data loaded successfully!', 'success');
                
            } catch (error) {
                showStatus(`Error loading data: ${error.message}`, 'error');
            }
        }

        async function runCompleteAnalysis() {
            const btn = document.getElementById('runAnalysisBtn');
            btn.disabled = true;
            btn.textContent = '⏳ Processing...';
            
            try {
                showStatus('Running complete analysis pipeline...', 'info');
                
                const response = await fetch('/api/process-all');
                const result = await response.json();
                
                if (result.error) {
                    throw new Error(result.error);
                }

                currentData = result;
                
                // Update all components
                updateStats(result.final_summary);
                updateMissingData(result.missing_data);
                updateCorrelation(result.correlation_data);
                updateOutlierAnalysis(result.outlier_info);
                updateFeatureEngineering(result.feature_info);
                
                // Load charts
                await loadPriceDistribution();
                await loadBoxPlot();
                await updateScatterPlot();
                
                showStatus('Analysis completed successfully!', 'success');
                
            } catch (error) {
                showStatus(`Error running analysis: ${error.message}`, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '🚀 Run Complete Analysis';
            }
        }

        function updateStats(summary) {
            document.getElementById('totalHouses').textContent = summary.total_houses;
            document.getElementById('avgPrice').textContent = summary.avg_price;
            document.getElementById('medianPrice').textContent = summary.median_price;
            document.getElementById('priceStd').textContent = summary.price_std;
            document.getElementById('statsGrid').style.display = 'grid';
        }

        function updateMissingData(missingData) {
            const container = document.getElementById('missingDataList');
            if (missingData.length === 0) {
                container.innerHTML = '<p>No missing data found!</p>';
                return;
            }
            
            let html = '<div style="display: grid; gap: 10px;">';
            missingData.slice(0, 10).forEach(item => {
                html += `
                    <div style="display: flex; justify-content: space-between; padding: 10px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #dc3545;">
                        <strong>${item.feature}</strong>
                        <span>${item.missing} missing (${item.percentage}%)</span>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        }

        function updateCorrelation(correlationData) {
            const container = document.getElementById('correlationList');
            if (!correlationData.top_features || correlationData.top_features.length === 0) {
                container.innerHTML = '<p>No correlation data available</p>';
                return;
            }
            
            let html = '<div style="display: grid; gap: 10px;">';
            correlationData.top_features.slice(0, 8).forEach(feature => {
                const width = Math.abs(feature.correlation) * 100;
                html += `