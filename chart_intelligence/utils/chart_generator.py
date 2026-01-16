import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime

class ChartGenerator:
    def __init__(self, df):
        self.df = df
        pio.templates.default = "plotly_dark"
    
    def create_comprehensive_chart(self, support_levels=None, resistance_levels=None):
        """Create multi-panel technical analysis chart"""
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2],  # Price, Volume, RSI, MACD
            subplot_titles=('Price with Indicators', 'Volume', 'RSI', 'MACD')
        )
        
        # 1. Price chart
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Add Moving Averages
        ma_cols = [col for col in self.df.columns if col.startswith('MA_')]
        for ma_col in ma_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[ma_col],
                    name=ma_col,
                    line=dict(width=1)
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands if available
        if 'BB_Upper' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['BB_Upper'],
                    name='BB Upper',
                    line=dict(dash='dash', color='gray'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['BB_Lower'],
                    name='BB Lower',
                    line=dict(dash='dash', color='gray'),
                    fill='tonexty',
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        # Add Support/Resistance lines
        if support_levels:
            for level in support_levels:
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="green",
                    opacity=0.5,
                    annotation_text=f"S: {level:.2f}",
                    row=1, col=1
                )
        
        if resistance_levels:
            for level in resistance_levels:
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="red",
                    opacity=0.5,
                    annotation_text=f"R: {level:.2f}",
                    row=1, col=1
                )
        
        # 2. Volume chart
        colors = ['red' if self.df['Close'].iloc[i] < self.df['Open'].iloc[i] 
                 else 'green' for i in range(len(self.df))]
        
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # 3. RSI chart
        if 'RSI' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # 4. MACD chart
        if 'MACD' in self.df.columns:
            # MACD line
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=4, col=1
            )
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['MACD_Signal'],
                    name='Signal',
                    line=dict(color='orange')
                ),
                row=4, col=1
            )
            # Histogram
            fig.add_trace(
                go.Bar(
                    x=self.df.index,
                    y=self.df['MACD_Diff'],
                    name='Histogram',
                    marker_color=['green' if x > 0 else 'red' 
                                for x in self.df['MACD_Diff']]
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Technical Analysis Dashboard',
            height=1200,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        return fig
    
    def save_chart(self, fig, filename='chart_output.html'):
        """Save chart as HTML (interactive)"""
        fig.write_html(f'outputs/{filename}')
        return f'outputs/{filename}'