import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

class SupportResistanceDetector:
    def __init__(self, df, order=5):
        self.df = df
        self.order = order  # Number of points to consider for local extrema
    
    def find_local_extrema(self, column='Close'):
        """Find local minima (support) and maxima (resistance)"""
        # Find local maxima
        max_idx = argrelextrema(
            self.df[column].values,
            np.greater,
            order=self.order
        )[0]
        
        # Find local minima
        min_idx = argrelextrema(
            self.df[column].values,
            np.less,
            order=self.order
        )[0]
        
        return {
            'resistance': self.df.iloc[max_idx][column].tolist(),
            'resistance_idx': max_idx.tolist(),
            'support': self.df.iloc[min_idx][column].tolist(),
            'support_idx': min_idx.tolist()
        }
    
    def calculate_pivot_points(self):
        """Calculate traditional pivot points"""
        if len(self.df) < 2:
            return {}
        
        recent = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        # Classic pivot points
        pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
        r1 = (2 * pivot) - prev['Low']
        s1 = (2 * pivot) - prev['High']
        r2 = pivot + (prev['High'] - prev['Low'])
        s2 = pivot - (prev['High'] - prev['Low'])
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2,
            's1': s1, 's2': s2
        }
    
    def find_key_levels(self, threshold=0.02):
        """Find significant support/resistance levels"""
        extrema = self.find_local_extrema()
        price_levels = {}
        
        # Combine all extremas
        all_levels = extrema['support'] + extrema['resistance']
        all_levels = sorted(all_levels)
        
        # Cluster nearby levels
        key_levels = []
        current_cluster = []
        
        for level in all_levels:
            if not current_cluster:
                current_cluster.append(level)
            elif abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < threshold:
                current_cluster.append(level)
            else:
                key_levels.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            key_levels.append(np.mean(current_cluster))
        
        # Classify as support or resistance based on current price
        current_price = self.df['Close'].iloc[-1]
        support = [l for l in key_levels if l < current_price]
        resistance = [l for l in key_levels if l > current_price]
        
        return {
            'support': sorted(support, reverse=True)[:5],  # Top 5 supports
            'resistance': sorted(resistance)[:5],  # Top 5 resistances
            'all_key_levels': key_levels
        }