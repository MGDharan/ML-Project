"""
FastAPI Backend for AI Stock Analysis
Educational purposes only - No financial advice
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import yfinance as yf
from datetime import datetime
import io
from PIL import Image
import re # Import regex module

app = FastAPI(
    title="Stock Analysis API",
    description="Educational stock market technical analysis API",
    version="1.0.0"
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class StockAnalysisRequest(BaseModel):
    symbol: str


class AnalysisResponse(BaseModel):
    detected_pattern: str
    trend: str  # Bullish / Bearish / Sideways
    option_bias: str  # CE / PE / No Trade
    buy_zone: str
    sell_zone: str
    risk_level: str
    disclaimer: str
    confidence: float
    chart_image_base64: Optional[str] = None


class LiveStockResponse(BaseModel):
    symbol: str
    current_price: float
    trend: str
    momentum: str  # RSI-style indicator
    option_bias: str
    risk_level: str
    disclaimer: str


# Pattern Detection Functions (Rule-based)
def detect_patterns(image: np.ndarray) -> dict:
    """
    Analyze chart image for technical patterns.
    Uses rule-based detection (educational placeholder).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours (potential patterns)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Simple pattern detection logic
    pattern_scores = {
        "Cup & Handle": 0.0,
        "Double Bottom": 0.0,
        "Double Top": 0.0,
        "Head & Shoulders": 0.0,
        "Breakout": 0.0,
        "Rejection": 0.0
    }
    
    # Analyze contour shapes (simplified pattern matching)
    if len(contours) > 0:
        # Check for U-shaped patterns (Cup & Handle, Double Bottom)
        for contour in contours[:5]:  # Check top 5 contours
            area = cv2.contourArea(contour)
            if area > 100:
                # Approximate shape
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # U-shape detection (simplified)
                if len(approx) < 10:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity < 0.7:  # Concave shape
                            pattern_scores["Cup & Handle"] += 0.2
                            pattern_scores["Double Bottom"] += 0.15
    
    # Trend detection from image
    height, width = gray.shape
    left_region = gray[:, :width//3]
    right_region = gray[:, 2*width//3:]
    
    left_mean = np.mean(left_region)
    right_mean = np.mean(right_region)
    
    # Determine trend
    if right_mean > left_mean * 1.05:
        trend = "Bullish"
        pattern_scores["Breakout"] += 0.3
    elif right_mean < left_mean * 0.95:
        trend = "Bearish"
        pattern_scores["Rejection"] += 0.3
        pattern_scores["Double Top"] += 0.2
    else:
        trend = "Sideways"
        pattern_scores["Head & Shoulders"] += 0.15
    
    # Get highest scoring pattern
    detected_pattern = max(pattern_scores, key=pattern_scores.get)
    confidence = min(pattern_scores[detected_pattern], 0.85)  # Cap at 85%
    
    return {
        "pattern": detected_pattern if pattern_scores[detected_pattern] > 0.1 else "No Clear Pattern",
        "trend": trend,
        "confidence": confidence
    }


def determine_option_bias(trend: str) -> str:
    """Determine options bias based on trend."""
    if trend == "Bullish":
        return "CE"
    elif trend == "Bearish":
        return "PE"
    else:
        return "No Trade"


def get_risk_level(pattern: str, trend: str) -> str:
    """Determine risk level based on pattern and trend."""
    high_risk_patterns = ["Rejection", "Double Top", "Head & Shoulders"]
    if pattern in high_risk_patterns or trend == "Sideways":
        return "High"
    elif trend in ["Bullish", "Bearish"]:
        return "Medium"
    return "Low"


@app.post("/api/analyze-charts", response_model=AnalysisResponse)
async def analyze_charts(
    chart_1day: UploadFile = File(...),
    chart_1year: UploadFile = File(...)
):
    """
    Analyze two chart images for technical patterns.
    Image 1: 1-day or intraday chart
    Image 2: 1-year chart for trend context
    """
    try:
        # Read images
        image1_bytes = await chart_1day.read()
        image2_bytes = await chart_1year.read()
        
        # Convert to numpy arrays
        img1 = np.array(Image.open(io.BytesIO(image1_bytes)))
        img2 = np.array(Image.open(io.BytesIO(image2_bytes)))
        
        # Convert RGB to BGR for OpenCV if needed
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        if len(img2.shape) == 3 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # Analyze short-term chart (1-day/intraday)
        analysis_1day = detect_patterns(img1)
        
        # Analyze long-term chart (1-year) for trend
        analysis_1year = detect_patterns(img2)
        
        # Combine analysis (prioritize 1-day pattern, 1-year trend)
        detected_pattern = analysis_1day["pattern"]
        trend = analysis_1year["trend"]  # Use long-term trend
        option_bias = determine_option_bias(trend)
        risk_level = get_risk_level(detected_pattern, trend)
        
        # Determine zones (textual, not price)
        if trend == "Bullish":
            buy_zone = "Near support levels or after minor pullback"
            sell_zone = "Near resistance levels or after significant gain"
        elif trend == "Bearish":
            buy_zone = "Avoid buying, wait for trend reversal confirmation"
            sell_zone = "Near resistance levels or bounce points"
        else:
            buy_zone = "Range-bound trading - buy near lower boundary"
            sell_zone = "Range-bound trading - sell near upper boundary"
        
        return AnalysisResponse(
            detected_pattern=detected_pattern,
            trend=trend,
            option_bias=option_bias,
            buy_zone=buy_zone,
            sell_zone=sell_zone,
            risk_level=risk_level,
            disclaimer="This analysis is for educational purposes only. Not financial advice. Market investments are subject to risk.",
            confidence=analysis_1day["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/api/live-stock", response_model=LiveStockResponse)
async def analyze_live_stock(request: StockAnalysisRequest):
    """
    Analyze live stock data using yfinance.
    Educational use only.
    """
    try:
        symbol = request.symbol.upper()

        # Basic input validation for stock symbol
        if not re.fullmatch(r'^[A-Z0-9.-]+$', symbol):
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol format: {symbol}. Please use a valid ticker like TATASTEEL or RELIANCE.")

        # Fetch stock data
        # Try with .NS suffix first for Indian stocks, then without
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="5d")

        if hist.empty:
            ticker = yf.Ticker(symbol) # Try without .NS suffix
            hist = ticker.history(period="5d")
        
        # Ensure we have at least 2 data points for price change calculation
        # and 'Close' column exists
        if hist.empty or len(hist) < 2 or 'Close' not in hist.columns:
            raise HTTPException(status_code=404, detail=f"Insufficient historical data for symbol: {symbol}. Please try a different symbol or check back later.")
        
        current_price = float(hist['Close'].iloc[-1])
        prev_price = float(hist['Close'].iloc[-2])

        # Safely get stock info (can sometimes fail or be incomplete)
        stock_info = {}
        try:
            stock_info = ticker.info
        except Exception as info_e:
            # Log this silently or with a warning if needed
            print(f"WARNING: Could not fetch detailed info for {symbol}: {info_e}")
        
        # Basic check for 'regularMarketPrice' as an alternative current price
        if 'regularMarketPrice' in stock_info and current_price == float(hist['Close'].iloc[-1]):
             current_price = stock_info['regularMarketPrice']
        
        # Calculate simple momentum (RSI-style)
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Determine trend
        if price_change > 1:
            trend = "Bullish"
            momentum = "Strong Upward"
        elif price_change > 0.3:
            trend = "Bullish"
            momentum = "Moderate Upward"
        elif price_change < -1:
            trend = "Bearish"
            momentum = "Strong Downward"
        elif price_change < -0.3:
            trend = "Bearish"
            momentum = "Moderate Downward"
        else:
            trend = "Sideways"
            momentum = "Neutral"
        
        option_bias = determine_option_bias(trend)
        
        # Risk level based on volatility
        if len(hist) >= 5:
            volatility = hist['Close'].pct_change().std() * 100
            if volatility > 3:
                risk_level = "High"
            elif volatility > 1.5:
                risk_level = "Medium"
            else:
                risk_level = "Low"
        else:
            risk_level = "Medium"
        
        return LiveStockResponse(
            symbol=symbol,
            current_price=round(current_price, 2),
            trend=trend,
            momentum=momentum,
            option_bias=option_bias,
            risk_level=risk_level,
            disclaimer="This analysis is for educational purposes only. Not financial advice. Market investments are subject to risk."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock analysis error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Stock Analysis API - Educational Use Only",
        "version": "1.0.0",
        "endpoints": {
            "/api/analyze-charts": "POST - Upload two chart images",
            "/api/live-stock": "POST - Analyze live stock by symbol"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
