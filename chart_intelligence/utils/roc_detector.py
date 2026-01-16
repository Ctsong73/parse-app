import pandas as pd
import matplotlib.pyplot as plt

def roc_trend_detector(series, window=3, threshold=0.02, consecutive=3):
    """
    Detects trends based on Rate of Change (ROC) with consecutive threshold logic.

    Parameters:
    - series (pd.Series): Series of closing prices.
    - window (int): Lookback period for ROC calculation.
    - threshold (float): Minimum ROC to qualify as a trend.
    - consecutive (int): Number of consecutive ROC values required to confirm a trend.

    Returns:
    - pd.DataFrame: DataFrame with Close, ROC, and Trend columns.
    """
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
    """
    Plots ROC values with trend coloring and threshold lines.

    Parameters:
    - df (pd.DataFrame): Output from roc_trend_detector.
    - threshold (float): Threshold used for trend detection.
    """
    roc_clean = df['ROC'].dropna()
    colors = df['Trend'].map({'Uptrend': 'green', 'Downtrend': 'red'}).fillna('Steelblue')
    colors = colors.loc[roc_clean.index]

    plt.figure(figsize=(10, 6))
    plt.bar(roc_clean.index, roc_clean * 100, color=colors)
    plt.axhline(y=threshold * 100, color='blue', linestyle='--', label='Uptrend Threshold')
    plt.axhline(y=-threshold * 100, color='red', linestyle='--', label='Downtrend Threshold')
    plt.title('Rate of Change (ROC) with Trend Detection')
    plt.xlabel('Time Index')
    plt.ylabel('ROC (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 📊 Sample Data

# Replace line 60 with full path
data = pd.read_excel("/home/cots/Clive/Python/meg.xlsx")
data.set_index('Date', inplace=True)

# 🚀 Run Analysis
threshold = 0.02
result = roc_trend_detector(data['Price'], window=7, threshold=threshold)
plot_roc_trends(result, threshold=threshold)

# 🖨️ Output Results
print(result)