import os
import openai
from typing import Dict, List, Optional
import json
import random

class ChartCommentator:
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize LLM commentator for chart analysis
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: gpt-4, gpt-3.5-turbo, claude-3-opus, etc.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.provider = "openai"  # or "anthropic", "gemini"
        
        if self.api_key and "openai" in self.model:
            openai.api_key = self.api_key
    
    def generate_commentary(self, 
                           chart_data: Dict,
                           market_context: str = "",
                           style: str = "professional") -> str:
        """
        Generate AI commentary on chart analysis
        
        Args:
            chart_data: Dictionary with technical analysis results
            market_context: News, earnings, macro context
            style: 'professional', 'casual', 'educational'
        """
        if not self.api_key:
            return self._get_fallback_commentary(chart_data)
        
        try:
            # Construct analysis prompt
            prompt = self._build_prompt(chart_data, market_context, style)
            
            if self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(style)},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                commentary = response.choices[0].message.content
            
            elif self.provider == "anthropic":
                # Claude API implementation
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                commentary = response.content[0].text
            
            return commentary.strip()
            
        except Exception as e:
            print(f"LLM API error: {e}")
            return self._get_fallback_commentary(chart_data)
    
    def _build_prompt(self, chart_data: Dict, context: str, style: str) -> str:
        """Build detailed prompt for LLM"""
        
        prompt = f"""
        Analyze this trading chart and provide {style} commentary:
        
        {self._format_chart_data(chart_data)}
        
        {f'Market Context: {context}' if context else ''}
        
        Please provide:
        1. Overall market sentiment
        2. Key technical observations  
        3. Support/Resistance analysis
        4. Risk assessment
        5. Trading outlook (1-2 weeks)
        
        Keep it concise and actionable.
        """
        return prompt
    
    def _format_chart_data(self, chart_data: Dict) -> str:
        """Format chart data for LLM consumption"""
        
        lines = ["CHART ANALYSIS DATA:"]
        
        if 'ticker' in chart_data:
            lines.append(f"Asset: {chart_data['ticker']}")
        
        if 'current_price' in chart_data:
            lines.append(f"Current Price: ${chart_data['current_price']:.2f}")
        
        if 'trend' in chart_data:
            lines.append(f"Trend: {chart_data['trend']}")
        
        if 'indicators' in chart_data:
            lines.append("\nTECHNICAL INDICATORS:")
            for ind, value in chart_data['indicators'].items():
                lines.append(f"  - {ind}: {value}")
        
        if 'levels' in chart_data and chart_data['levels']:
            lines.append("\nSUPPORT/RESISTANCE LEVELS:")
            for level in chart_data['levels']:
                lines.append(f"  - {level['type'].upper()} at ${level['price']:.2f} "
                           f"(Strength: {level['strength']:.2f})")
        
        if 'patterns' in chart_data and chart_data['patterns']:
            lines.append(f"\nCHART PATTERNS: {', '.join(chart_data['patterns'])}")
        
        if 'volatility' in chart_data:
            lines.append(f"\nVolatility: {chart_data['volatility']:.2%}")
        
        return "\n".join(lines)
    
    def _get_system_prompt(self, style: str) -> str:
        """Get system prompt based on style"""
        
        prompts = {
            "professional": """You are a professional trading analyst at a major investment bank. 
            Provide clear, data-driven analysis. Be conservative with predictions.
            Focus on risk management and probabilities.""",
            
            "casual": """You are a helpful trading assistant. Explain things simply.
            Use analogies and everyday language. Make trading accessible to beginners.""",
            
            "educational": """You are a trading educator. Explain not just what but WHY.
            Teach technical analysis concepts as you analyze. Include learning points."""
        }
        
        return prompts.get(style, prompts["professional"])
    
    def _get_fallback_commentary(self, chart_data: Dict) -> str:
        """Fallback commentary when API is unavailable"""
        
        ticker = chart_data.get('ticker', 'the asset')
        trend = chart_data.get('trend', 'neutral').lower()
        
        # Different templates for different trend types
        if trend == "bullish":
            commentaries = [
                f"Based on the analysis, {ticker} shows bullish momentum with prices trending upward. "
                "Consider looking for buying opportunities near support levels with proper risk management.",
                
                f"The chart displays bullish characteristics for {ticker}. "
                "Monitor for breakout opportunities above resistance levels while maintaining stop losses.",
                
                f"Technical indicators suggest upward bias in {ticker}. "
                "The risk-reward ratio appears favorable for long positions with tight risk controls."
            ]
        elif trend == "bearish":
            commentaries = [
                f"Based on the analysis, {ticker} shows bearish momentum with prices trending downward. "
                "Consider looking for selling opportunities near resistance levels with proper risk management.",
                
                f"The chart displays bearish characteristics for {ticker}. "
                "Monitor for breakdown opportunities below support levels while maintaining stop losses.",
                
                f"Technical indicators suggest downward bias in {ticker}. "
                "The risk-reward ratio appears favorable for short positions with tight risk controls."
            ]
        else:
            commentaries = [
                f"Based on the analysis, {ticker} shows neutral momentum with prices consolidating. "
                "Consider range-trading strategies between support and resistance levels.",
                
                f"The chart displays range-bound characteristics for {ticker}. "
                "Monitor for breakout opportunities in either direction with confirmation signals.",
                
                f"Technical indicators suggest neutral bias in {ticker}. "
                "Wait for clearer directional signals before committing to positions."
            ]
        
        return random.choice(commentaries)
    
    def ask_followup(self, question: str, previous_context: Dict) -> str:
        """Allow follow-up questions about the analysis"""
        
        prompt = f"""
        User follow-up question: {question}
        
        Previous analysis context:
        {json.dumps(previous_context, indent=2)}
        
        Answer the question specifically based on the analysis above.
        If you don't have enough information, say so.
        """
        
        try:
            if self.api_key and self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Answer trading questions based on provided analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Follow-up API error: {e}")
        
        return "I cannot answer follow-up questions without API access. Please review the analysis above."


# Example usage
if __name__ == "__main__":
    # Initialize with API key from environment
    commentator = ChartCommentator(model="gpt-3.5-turbo")
    
    # Sample chart data
    sample_data = {
        'ticker': 'AAPL',
        'current_price': 175.25,
        'trend': 'bullish',
        'indicators': {
            'RSI': 62.5,
            'MACD': 'bullish crossover',
            'Volume': 'above average'
        },
        'levels': [
            {'type': 'support', 'price': 170.50, 'strength': 0.8},
            {'type': 'resistance', 'price': 180.00, 'strength': 0.7}
        ],
        'volatility': 0.18
    }
    
    # Generate commentary
    commentary = commentator.generate_commentary(
        sample_data, 
        market_context="Upcoming earnings announcement next week",
        style="professional"
    )
    
    print("=" * 60)
    print("AI Commentary:")
    print("=" * 60)
    print(commentary)
    print("\n" + "=" * 60)
    print("✅ ChartCommentator is ready!")
    print("=" * 60)

    