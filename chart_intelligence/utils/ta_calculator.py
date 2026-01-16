import pandas as pd
import numpy as np
import ta

class TACalculator:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        macd_indicator = ta.trend.MACD(
            close=self.df['Close'],
            window_slow=slow,
            window_fast=fast,
            window_sign=signal
        )
        self.df['MACD'] = macd_indicator.macd()
        self.df['MACD_Signal'] = macd_indicator.macd_signal()
        self.df['MACD_Diff'] = macd_indicator.macd_diff()
        return self.df[['MACD', 'MACD_Signal', 'MACD_Diff']]
    
    def calculate_rsi(self, period=14):
        """Calculate RSI indicator"""
        self.df['RSI'] = ta.momentum.RSIIndicator(
            close=self.df['Close'],
            window=period
        ).rsi()
        return self.df['RSI']
    
    def calculate_moving_averages(self, periods=[20, 50, 200]):
        """Calculate multiple moving averages"""
        for period in periods:
            col_name = f'MA_{period}'
            self.df[col_name] = ta.trend.sma_indicator(
                close=self.df['Close'],
                window=period
            )
        return self.df[[f'MA_{p}' for p in periods]]
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        bb = ta.volatility.BollingerBands(
            close=self.df['Close'],
            window=period,
            window_dev=std_dev
        )
        self.df['BB_Upper'] = bb.bollinger_hband()
        self.df['BB_Middle'] = bb.bollinger_mavg()
        self.df['BB_Lower'] = bb.bollinger_lband()
        self.df['BB_Width'] = bb.bollinger_wband()
        return self.df[['BB_Upper', 'BB_Middle', 'BB_Lower']]
    
    # In chart_intelligence/utils/ta_calculator.py, add:

class TACalculator:
    def __init__(self, df):
        self.df = df.copy()
    
    # ... existing methods ...
    
    def calculate_roc(self, window=7, threshold=0.01, consecutive=3):
        """Calculate Rate of Change trend detection"""
        from .roc_detector import ROCTrendDetector
        
        detector = ROCTrendDetector(
            series=self.df['Close'],
            window=window,
            threshold=threshold,
            consecutive=consecutive
        )
        
        roc_result = detector.detect()
        
        # Add ROC columns to DataFrame
        self.df['ROC'] = roc_result['ROC']
        self.df['ROC_Percent'] = roc_result['ROC_Percent']
        self.df['ROC_Trend'] = roc_result['Trend']
        self.df['ROC_Signal'] = roc_result['Signal']
        
        return {
            'data': roc_result,
            'summary': detector.get_summary_stats(),
            'periods': detector.get_trend_periods()
        }
    
    def calculate_all_indicators(self, roc_params=None):
        """Calculate all technical indicators including ROC"""
        # Existing calculations
        self.calculate_macd()
        self.calculate_rsi()
        self.calculate_moving_averages()
        self.calculate_bollinger_bands()
        
        # Add ROC
        if roc_params is None:
            roc_params = {'window': 7, 'threshold': 0.01, 'consecutive': 3}
        
        roc_results = self.calculate_roc(**roc_params)
        
        return {
            'df': self.df,
            'roc_results': roc_results
        }
    
    