import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import tempfile
import os
import sys
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime

# Set page config FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="Parse: Technical Analysis AI Assistant",
    page_icon="📈",
    layout="wide"
)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ROC Detector functions (inlined)
def roc_trend_detector(series, window=3, threshold=0.02, consecutive=3):
    """Detects trends based on Rate of Change (ROC)"""
    roc = series.pct_change(periods=window)
    trend = pd.Series("-", index=series.index)

    up_mask = roc > threshold
    down_mask = roc < -threshold

    up_count = up_mask.rolling(window=consecutive).sum()
    down_count = down_mask.rolling(window=consecutive).sum()

    trend.loc[up_count[up_count >= consecutive].index] = "Uptrend"
    trend.loc[down_count[down_count >= consecutive].index] = "Downtrend"

    return pd.DataFrame({
        'Close': series,
        'ROC': roc,
        'Trend': trend
    })

def plot_roc_trends(df, threshold=0.01):
    """Plots ROC values with trend coloring"""
    roc_clean = df['ROC'].dropna()
    colors = df['Trend'].map({'Uptrend': 'green', 'Downtrend': 'red'}).fillna('Steelblue')
    colors = colors.loc[roc_clean.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(roc_clean.index, roc_clean * 100, color=colors)
    ax.axhline(y=threshold * 100, color='blue', linestyle='--', label='Uptrend Threshold')
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Also hide any GitHub links in text */
    a[href*="github.com"],
    a[href*="github.io"] {
        display: none !important;
    }
    plt.tight_layout()
    return fig

# Initialize ROC availability
ROC_AVAILABLE = True

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: normal;
    }
    /* Hide GitHub icon - Comprehensive selectors */
    [data-testid="stMainMenu"],
    [data-testid="baseButton-header"],
    button[data-testid="baseButton-headerNoPadding"],
    .st-emotion-cache-1v0mbdj,
    .st-emotion-cache-1inwz65,
    [href*="github"],
    a[href*="github"],
    svg[data-testid*="github"],
    button[aria-label*="GitHub"],
    button[title*="GitHub"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
    }
    
    /* Hide GitHub links */
    a[href*="github.com"],
    a[href*="github.io"] {
        display: none !important;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .trend-up { color: #10B981; font-weight: bold; }
    .trend-down { color: #EF4444; font-weight: bold; }
    .trend-neutral { color: #6B7280; font-weight: bold; }
    .ai-analysis {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4f46e5;
        margin: 20px 0;
        font-family: 'Georgia', serif;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .instructions-box {
        background-color: #f0f9ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0369a1;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title with different font sizes
st.markdown("<h1 class='main-header'>Parse</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Technical Analysis AI Assistant</h2>", unsafe_allow_html=True)
st.markdown("Upload your price data for comprehensive technical analysis with AI insights.")

# ----------------------------------------------------
# GROQ API SETUP - FREE & EASY
# ----------------------------------------------------
def setup_groq():
    """Setup Groq API - FREE & No Credit Card Needed"""
    if 'groq_api_key' not in st.session_state:
        # Try to get from secrets first
        try:
            from streamlit import secrets
            groq_key = secrets.get("GROQ_API_KEY", "")
        except:
            groq_key = ""
        
        if groq_key:
            st.session_state.groq_api_key = groq_key
            st.session_state.groq_configured = True
            return groq_key
        else:
            # Show setup instructions in sidebar
            with st.sidebar.expander("🔑 AI API Setup", expanded=True):
                st.markdown("""
                ### Configure AI Analysis:
                
                1. **Get FREE API Key:**
                   - Visit [console.groq.com](https://console.groq.com)
                   - Sign up (Google/GitHub)
                   - Copy your API key
                
                2. **Add to Streamlit:**
                   ```toml
                   # .streamlit/secrets.toml
                   GROQ_API_KEY = "your-key-here"
                   ```
                
                **FREE Tier:** 30 requests/minute
                """)
                
                # Allow manual input for testing
                test_key = st.text_input("Or enter key for this session:", type="password")
                if test_key:
                    st.session_state.groq_api_key = test_key
                    st.session_state.groq_configured = True
                    st.success("✅ Using session key")
                    return test_key
            
            st.session_state.groq_configured = False
            return None
    
    return st.session_state.get('groq_api_key')

# Check if Groq is configured
GROQ_CONFIGURED = setup_groq() is not None

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    # Data frequency selection
    st.subheader("📅 Data Frequency")
    data_frequency = st.selectbox(
        "Select data frequency",
        ["Daily", "Weekly"],
        help="Select whether your data is daily or weekly prices"
    )
    
    # Determine MA periods based on frequency
    if data_frequency == "Daily":
        ma_periods = [10, 50]
        ma_default = [10, 50]
    else:  # Weekly
        ma_periods = [10, 20]
        ma_default = [10, 20]
    
    st.subheader("📊 Moving Average Periods")
    selected_ma_periods = st.multiselect(
        "Select MA periods to display",
        ma_periods,
        default=ma_default
    )
    
    st.subheader("📈 Chart Options")
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
    show_ma_chart = st.checkbox("Show Moving Averages Chart", value=True)
    
    if ROC_AVAILABLE:
        st.subheader("🎯 ROC Trend Detection")
        show_roc = st.checkbox("Show ROC Analysis", value=True)
        roc_window = st.slider("ROC Window", 3, 21, 7, help="Lookback period for ROC calculation")
        roc_threshold = st.slider("ROC Threshold (%)", 0.1, 5.0, 2.0, step=0.1) / 100  # Default 2%
        roc_consecutive = st.slider("Consecutive Periods", 1, 7, 3, help="Number of consecutive periods to confirm trend")
    
    st.subheader("🤖 AI Analysis Settings")
    
    if not GROQ_CONFIGURED:
        st.error("❌ Groq API key not configured")
        st.info("Configure AI key in the expander above")
        enable_ai_analysis = False
    else:
        enable_ai_analysis = st.checkbox("Enable AI Analysis", value=True, 
                                         help="Generate detailed analysis using AI (FREE with Groq)")
    
    if enable_ai_analysis and GROQ_CONFIGURED:
        ai_provider = st.selectbox("AI Model", [
            "llama-3.1-8b-instant",      # Fast & efficient
            "llama-3.3-70b-versatile",   # More powerful
            "gemma2-9b-it",              # Good alternative
        ])
        ai_temperature = st.slider("AI Creativity", 0.0, 1.0, 0.7, 
                                   help="Higher values = more creative, Lower values = more focused")
    
    st.divider()
    st.info("📋 Required columns: Date, Price (or Close)")

# ----------------------------------------------------
# DATA PROCESSING FUNCTIONS
# ----------------------------------------------------
def load_data_from_upload(uploaded_file):
    """Load data from uploaded file"""
    try:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.temp') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(temp_path)
        else:  # Excel
            df = pd.read_excel(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Find date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
        else:
            st.warning("No date column found. Using index as date.")
        
        # Find price column
        column_mapping = {}
        price_col_found = False
        
        for col in df.columns:
            col_lower = col.lower()
            if 'close' in col_lower or 'price' in col_lower or 'value' in col_lower:
                column_mapping[col] = 'Close'
                price_col_found = True
                break
        
        if not price_col_found:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    column_mapping[col] = 'Close'
                    price_col_found = True
                    break
        
        if not price_col_found:
            st.error("No price column found.")
            return None
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Create OHLC from Close price
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
        
        # Check data sufficiency
        if data_frequency == "Daily" and len(df) < 250:
            st.warning(f"⚠️ For daily data, 1 year (250+ trading days) is recommended for accurate analysis. You have {len(df)} days.")
        elif data_frequency == "Weekly" and len(df) < 156:
            st.warning(f"⚠️ For weekly data, 3 years (156+ weeks) is recommended for accurate analysis. You have {len(df)} weeks.")
        
        st.info(f"✅ Data loaded: {len(df)} {data_frequency.lower()} rows")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def calculate_indicators(df, ma_periods=[10, 50]):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving Averages
    for period in ma_periods:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def find_support_resistance(df, lookback=20):
    """Find support and resistance levels"""
    if len(df) < lookback:
        return [], []
    
    highs = df['High'].rolling(window=lookback, center=True).max()
    lows = df['Low'].rolling(window=lookback, center=True).min()
    
    recent_highs = highs.iloc[-lookback:].unique()
    recent_lows = lows.iloc[-lookback:].unique()
    
    current_price = df['Close'].iloc[-1]
    
    support_levels = sorted([l for l in recent_lows if l < current_price], reverse=True)[:3]
    resistance_levels = sorted([h for h in recent_highs if h > current_price])[:3]
    
    return support_levels, resistance_levels

def generate_summary(df, support_levels, resistance_levels):
    """Generate trading summary"""
    if len(df) < 2:
        return {
            'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
            'price_change': 0,
            'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else None,
            'trend': 'Unknown',
            'nearest_support': None,
            'nearest_resistance': None
        }
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Determine trend
    if len(df) >= 5:
        if latest['Close'] > df['Close'].iloc[-5]:
            trend = 'Uptrend'
        else:
            trend = 'Downtrend'
    else:
        trend = 'Unknown'
    
    summary = {
        'current_price': latest['Close'],
        'price_change': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
        'rsi': latest['RSI'] if 'RSI' in df.columns else None,
        'trend': trend,
        'nearest_support': support_levels[0] if support_levels else None,
        'nearest_resistance': resistance_levels[0] if resistance_levels else None
    }
    
    return summary

# ----------------------------------------------------
# CHART FUNCTIONS
# ----------------------------------------------------
def create_main_chart(df, support_levels, resistance_levels):
    """Create main price chart"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Price',
            line=dict(color='#1E3A8A', width=2),
            mode='lines'
        )
    )
    
    for level in support_levels:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color="green",
            opacity=0.5,
            annotation_text=f"S: {level:.2f}"
        )
    
    for level in resistance_levels:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color="red",
            opacity=0.5,
            annotation_text=f"R: {level:.2f}"
        )
    
    fig.update_layout(
        title=f'Price Chart ({data_frequency})',
        height=400,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Price",
        template='plotly_white'
    )
    
    return fig

def create_ma_chart(df, selected_ma_periods):
    """Create Moving Averages chart"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Price',
            line=dict(color='#1E3A8A', width=1.5),
            mode='lines',
            opacity=0.7
        )
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    for i, period in enumerate(selected_ma_periods):
        ma_col = f'MA_{period}'
        if ma_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma_col],
                    name=f'MA {period}',
                    line=dict(color=colors[i % len(colors)], width=2)
                )
            )
    
    fig.update_layout(
        title='Moving Averages',
        height=400,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Price",
        template='plotly_white'
    )
    
    return fig

def create_bollinger_chart(df):
    """Create Bollinger Bands chart"""
    if 'BB_Upper' not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Price',
            line=dict(color='#1E3A8A', width=2),
            mode='lines'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash', width=1),
            opacity=0.5
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Middle'],
            name='BB Middle',
            line=dict(color='gray', width=1),
            opacity=0.5
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash', width=1),
            fill='tonexty',
            opacity=0.3
        )
    )
    
    fig.update_layout(
        title='Bollinger Bands',
        height=400,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Price",
        template='plotly_white'
    )
    
    return fig

def create_rsi_chart(df):
    """Create RSI chart"""
    if 'RSI' not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='#8E44AD', width=2)
        )
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        height=300,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        template='plotly_white'
    )
    
    return fig

def create_macd_chart(df):
    """Create MACD chart"""
    if 'MACD' not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD Line',
            line=dict(color='#3498DB', width=2),
            mode='lines'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            name='Signal Line',
            line=dict(color='#E74C3C', width=2),
            mode='lines'
        )
    )
    
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['MACD_Hist'],
            name='MACD Histogram',
            marker_color=colors,
            opacity=0.6,
            yaxis='y2'
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Moving Average Convergence Divergence (MACD)',
        height=400,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="MACD & Signal Line",
        yaxis2=dict(
            title="Histogram",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template='plotly_white'
    )
    
    return fig

# ----------------------------------------------------
# GROQ AI ANALYSIS FUNCTION (UPDATED MODELS)
# ----------------------------------------------------
def generate_groq_analysis(summary, df, support_levels, resistance_levels, selected_ma_periods, data_frequency):
    """Generate AI analysis using Groq API (FREE)"""
    
    # Get API key
    api_key = st.session_state.get('groq_api_key')
    if not api_key:
        try:
            from streamlit import secrets
            api_key = secrets.get("GROQ_API_KEY", "")
        except:
            api_key = ""
    
    if not api_key:
        return "⚠️ Please configure Groq API key first. Get FREE key at: console.groq.com"
    
    # Prepare the prompt
    latest_data = df.iloc[-1] if len(df) > 0 else {}
    
    analysis_prompt = f"""
    Generate a comprehensive 500-word technical analysis report:
    
    SECURITY ANALYSIS REPORT
    ========================
    
    DATA SUMMARY:
    - Data Frequency: {data_frequency}
    - Total Data Points: {len(df)}
    - Time Period: {df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'} to {df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'}
    
    CURRENT MARKET CONDITIONS:
    - Current Price: ${summary.get('current_price', 0):.2f}
    - Price Change: {summary.get('price_change', 0):+.2f}%
    - Market Trend: {summary.get('trend', 'Unknown')}
    - RSI Level: {summary.get('rsi', 0):.2f} ({'Overbought' if summary.get('rsi', 0) > 70 else 'Oversold' if summary.get('rsi', 0) < 30 else 'Neutral'})
    
    TECHNICAL INDICATORS:
    - Moving Averages: {', '.join([f'MA {period}' for period in selected_ma_periods])}
    - MACD: {latest_data.get('MACD', 0):.4f}
    - MACD Signal: {latest_data.get('MACD_Signal', 0):.4f}
    
    KEY LEVELS:
    - Support: {', '.join([f'${level:.2f}' for level in support_levels[:3]]) if support_levels else 'None'}
    - Resistance: {', '.join([f'${level:.2f}' for level in resistance_levels[:3]]) if resistance_levels else 'None'}
    
    Please provide a detailed technical analysis covering:
    1. Trend analysis and momentum
    2. Key support/resistance levels
    3. Technical indicator signals
    4. Trading recommendations with specific price levels
    5. Risk management advice
    
    Be professional, concise, and actionable. Focus on practical trading insights.
    """
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use updated model names
        data = {
            "model": ai_provider if 'ai_provider' in locals() else "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a professional technical analyst at an investment bank. Provide precise, data-driven analysis with actionable insights."},
                {"role": "user", "content": analysis_prompt}
            ],
            "temperature": ai_temperature if 'ai_temperature' in locals() else 0.7,
            "max_tokens": 800
        }
        
        with st.spinner("🤖 AI is analyzing the data..."):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                return analysis
            elif response.status_code == 401:
                return "❌ Invalid API key. Please check your Groq API key at console.groq.com"
            elif response.status_code == 429:
                return "⚠️ Rate limit exceeded. Free tier allows 30 requests/minute. Please wait a moment."
            else:
                # Try to parse error message
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg = error_data['error'].get('message', 'Unknown error')
                        if 'decommissioned' in error_msg:
                            return f"❌ Model error: {error_msg}\n\nPlease select a different model from the sidebar."
                        return f"❌ API Error: {error_msg}"
                except:
                    pass
                return f"❌ API Error {response.status_code}. Please try again."
                
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ----------------------------------------------------
# ROC ANALYSIS FUNCTIONS
# ----------------------------------------------------
def get_roc_summary_stats(roc_results):
    """Get summary statistics from ROC results"""
    if roc_results is None or roc_results.empty:
        return {
            'current_trend': 'No Data',
            'current_roc': 0,
            'current_signal': 'NEUTRAL',
            'uptrend_count': 0,
            'downtrend_count': 0,
            'avg_roc_uptrend': 0,
            'avg_roc_downtrend': 0
        }
    
    current_trend = roc_results['Trend'].iloc[-1] if not roc_results['Trend'].iloc[-1] == '-' else 'No Trend'
    current_roc = roc_results['ROC'].iloc[-1] * 100 if not pd.isna(roc_results['ROC'].iloc[-1]) else 0
    
    if current_trend == 'Uptrend':
        current_signal = 'BUY'
    elif current_trend == 'Downtrend':
        current_signal = 'SELL'
    else:
        current_signal = 'NEUTRAL'
    
    uptrend_count = (roc_results['Trend'] == 'Uptrend').sum()
    downtrend_count = (roc_results['Trend'] == 'Downtrend').sum()
    
    avg_roc_uptrend = roc_results.loc[roc_results['Trend'] == 'Uptrend', 'ROC'].mean() * 100 if uptrend_count > 0 else 0
    avg_roc_downtrend = roc_results.loc[roc_results['Trend'] == 'Downtrend', 'ROC'].mean() * 100 if downtrend_count > 0 else 0
    
    return {
        'current_trend': current_trend,
        'current_roc': current_roc,
        'current_signal': current_signal,
        'uptrend_count': uptrend_count,
        'downtrend_count': downtrend_count,
        'avg_roc_uptrend': avg_roc_uptrend,
        'avg_roc_downtrend': avg_roc_downtrend
    }

def display_roc_analysis(df, roc_window, roc_threshold, roc_consecutive):
    """Display ROC Trend Detection analysis"""
    if not ROC_AVAILABLE:
        return None
    
    try:
        roc_results = roc_trend_detector(
            series=df['Close'],
            window=roc_window,
            threshold=roc_threshold,
            consecutive=roc_consecutive
        )
        
        summary = get_roc_summary_stats(roc_results)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trend_class = "trend-up" if summary['current_trend'] == 'Uptrend' else "trend-down" if summary['current_trend'] == 'Downtrend' else "trend-neutral"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Current Trend</h3>
                <p class='{trend_class}' style='font-size: 24px;'>{summary['current_trend']}</p>
                <p>ROC: {summary['current_roc']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            signal_color = "trend-up" if summary['current_signal'] == 'BUY' else "trend-down" if summary['current_signal'] == 'SELL' else "trend-neutral"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Current Signal</h3>
                <p class='{signal_color}' style='font-size: 24px;'>{summary['current_signal']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Uptrend Periods</h3>
                <p style='font-size: 24px; color: #10B981;'>{summary['uptrend_count']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Downtrend Periods</h3>
                <p style='font-size: 24px; color: #EF4444;'>{summary['downtrend_count']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Plot ROC chart
        st.subheader("ROC Analysis Chart")
        roc_fig = plot_roc_trends(roc_results, threshold=roc_threshold)
        st.pyplot(roc_fig)
        
        return roc_results
        
    except Exception as e:
        st.error(f"Error in ROC analysis: {str(e)}")
        return None

# ----------------------------------------------------
# FALLBACK ANALYSIS (if API fails)
# ----------------------------------------------------
def generate_fallback_analysis(summary, support_levels, resistance_levels):
    """Generate basic analysis if API fails"""
    
    if summary['price_change'] > 2:
        sentiment = "BULLISH 📈"
        recommendation = "Consider buying on dips with stop-loss below nearest support."
        outlook = "Positive momentum suggests further upside potential."
    elif summary['price_change'] < -2:
        sentiment = "BEARISH 📉"
        recommendation = "Consider selling on rallies or wait for confirmation of support."
        outlook = "Downward pressure suggests caution is warranted."
    else:
        sentiment = "NEUTRAL ↔️"
        recommendation = "Range-bound trading likely. Buy near support, sell near resistance."
        outlook = "Consolidation phase; wait for breakout confirmation."
    
    support_text = ', '.join([f'${level:.2f}' for level in support_levels[:3]]) if support_levels else 'None identified'
    resistance_text = ', '.join([f'${level:.2f}' for level in resistance_levels[:3]]) if resistance_levels else 'None identified'
    
    return f"""
    📊 TECHNICAL ANALYSIS REPORT
    
    MARKET SENTIMENT: {sentiment}
    
    CURRENT PRICE: ${summary['current_price']:.2f}
    DAILY CHANGE: {summary['price_change']:+.2f}%
    MARKET TREND: {summary['trend']}
    
    KEY TECHNICAL LEVELS:
    • Support: {support_text}
    • Resistance: {resistance_text}
    
    TRADING RECOMMENDATION:
    {recommendation}
    
    RISK MANAGEMENT:
    • Stop-loss: Place 3-5% below entry price
    • Take-profit: Target 5-8% above entry
    • Position size: Limit to 2-5% of total portfolio
    
    MARKET OUTLOOK:
    {outlook}
    
    NOTE: This is a basic analysis. For detailed AI-powered insights, ensure your Groq API is properly configured.
    """

# ----------------------------------------------------
# MAIN APP
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload your price data file",
    type=['csv', 'xlsx', 'xls'],
    help="File must contain Date and Price columns"
)

if uploaded_file is not None:
    with st.spinner("📊 Loading and analyzing data..."):
        df = load_data_from_upload(uploaded_file)
        
        if df is not None:
            df = calculate_indicators(df, selected_ma_periods)
            support_levels, resistance_levels = find_support_resistance(df)
            summary = generate_summary(df, support_levels, resistance_levels)
            
            # Create tabs
            tab_names = ["📈 Main Analysis", "📊 Data Preview"]
            if ROC_AVAILABLE and show_roc:
                tab_names.append("🎯 ROC Analysis")
            if enable_ai_analysis:
                tab_names.append("🤖 AI Analysis")
            
            tabs = st.tabs(tab_names)
            
            with tabs[0]:
                st.subheader("Price Chart")
                main_fig = create_main_chart(df, support_levels, resistance_levels)
                st.plotly_chart(main_fig, use_container_width=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${summary['current_price']:.2f}", f"{summary['price_change']:+.2f}%")
                with col2:
                    if summary['rsi']:
                        rsi_status = "Overbought" if summary['rsi'] > 70 else "Oversold" if summary['rsi'] < 30 else "Neutral"
                        st.metric("RSI", f"{summary['rsi']:.2f}", rsi_status)
                with col3:
                    st.metric("Trend", summary['trend'])
                
                # Key Levels
                col4, col5 = st.columns(2)
                with col4:
                    st.subheader("🛡️ Support Levels")
                    if support_levels:
                        for i, level in enumerate(support_levels[:3], 1):
                            st.write(f"**S{i}:** ${level:.2f}")
                    else:
                        st.write("No support levels found")
                
                with col5:
                    st.subheader("⛰️ Resistance Levels")
                    if resistance_levels:
                        for i, level in enumerate(resistance_levels[:3], 1):
                            st.write(f"**R{i}:** ${level:.2f}")
                    else:
                        st.write("No resistance levels found")
                
                # Additional charts
                if show_ma_chart:
                    st.subheader("Moving Averages")
                    ma_fig = create_ma_chart(df, selected_ma_periods)
                    st.plotly_chart(ma_fig, use_container_width=True)
                
                if show_bollinger:
                    st.subheader("Bollinger Bands")
                    bb_fig = create_bollinger_chart(df)
                    if bb_fig:
                        st.plotly_chart(bb_fig, use_container_width=True)
                
                if show_rsi:
                    st.subheader("RSI")
                    rsi_fig = create_rsi_chart(df)
                    if rsi_fig:
                        st.plotly_chart(rsi_fig, use_container_width=True)
                
                if show_macd:
                    st.subheader("MACD")
                    macd_fig = create_macd_chart(df)
                    if macd_fig:
                        st.plotly_chart(macd_fig, use_container_width=True)
            
            with tabs[1]:
                st.subheader("Data Preview")
                st.dataframe(df.tail(50), use_container_width=True)
                
                csv = df.to_csv()
                st.download_button(
                    "📥 Download Data",
                    data=csv,
                    file_name="ta_data.csv",
                    mime="text/csv"
                )
            
            if ROC_AVAILABLE and show_roc and len(tabs) > 2 and "🎯 ROC Analysis" in tab_names:
                with tabs[tab_names.index("🎯 ROC Analysis")]:
                    display_roc_analysis(df, roc_window, roc_threshold, roc_consecutive)
            
            if enable_ai_analysis and "🤖 AI Analysis" in tab_names:
                with tabs[tab_names.index("🤖 AI Analysis")]:
                    st.header("🤖 AI-Powered Analysis")
                    
                    if st.button("Generate AI Analysis", type="primary"):
                        # Try Groq API first
                        analysis = generate_groq_analysis(
                            summary, df, support_levels, resistance_levels, 
                            selected_ma_periods, data_frequency
                        )
                        
                        # Check if API failed
                        if analysis.startswith("❌") or analysis.startswith("⚠️"):
                            st.warning("Using fallback analysis. Check API configuration for full AI features.")
                            analysis = generate_fallback_analysis(summary, support_levels, resistance_levels)
                        
                        st.markdown(f"""
                        <div class='ai-analysis'>
                            {analysis}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download option
                        st.download_button(
                            "📝 Download Analysis Report",
                            data=analysis,
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            
            st.success("✅ Analysis complete!")

else:
    st.info("👆 Upload a file to begin")
    
    # OPERATING INSTRUCTIONS SECTION
    with st.expander("📋 Operating Instructions", expanded=True):
        st.markdown("""
        <div class='instructions-box'>
        <h3>📊 Data Requirements for Accurate Analysis:</h3>
        
        **For Daily Data:**
        • Upload **1 year or more** of daily price data (250+ trading days recommended)
        • Ensures reliable moving averages and trend analysis
        • Provides sufficient data for technical indicators
        
        **For Weekly Data:**
        • Upload **3 years or more** of weekly price data (156+ weeks recommended)
        • Needed for accurate long-term trend identification
        • Reduces noise and false signals
        
        **File Format:**
        • CSV or Excel format
        • Required columns: <strong>Date</strong> and <strong>Price</strong> (or Close)
        • Date column should be in proper date format
        
        **Optimal Results:**
        • More data = Better analysis accuracy
        • Clean, consistent data without gaps
        • Regular trading intervals (daily/weekly)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sample data
        st.write("**Sample Data Format:**")
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Price': [102.00, 103.50, 101.80, 105.00, 106.20]
        })
        st.dataframe(sample_data)
        
        st.download_button(
            "📥 Download Sample Data",
            data=sample_data.to_csv(index=False),
            file_name="sample_price_data.csv",
            mime="text/csv"
        )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p style='font-size: 16px; font-weight: bold;'>Parse: Technical Analysis AI Assistant</p>
    <p style='font-size: 12px;'>Powered by Groq AI • Rate of Change (ROC) Trend Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)