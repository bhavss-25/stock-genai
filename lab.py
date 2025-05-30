import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🤖 Stock Price Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        margin: 20px 0;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### 📈 Enter OHLC values to predict next day's closing price")

@st.cache_data
def load_and_prepare_data():
    """Load and prepare training data"""
    try:
        # Load the data
        stock_data = pd.read_csv(r"C:\Users\bhavya\Desktop\Code\Python\stock.csv")
        
        # Handle different date formats
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
        stock_data = stock_data.dropna(subset=['Date'])
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        
        # Handle scientific notation in Volume
        if stock_data['Volume'].dtype == 'object':
            stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
        
        # Remove NaN values
        stock_data = stock_data.dropna()
        
        # Create features for predicting next day's close
        # We'll use current day's OHLC + previous day's close to predict next day's close
        stock_data['Prev_Close'] = stock_data['Close'].shift(1)
        stock_data['Next_Close'] = stock_data['Close'].shift(-1)  # Target variable
        
        # Additional technical features
        stock_data['Price_Range'] = stock_data['High'] - stock_data['Low']
        stock_data['Body_Size'] = abs(stock_data['Close'] - stock_data['Open'])
        stock_data['Upper_Shadow'] = stock_data['High'] - np.maximum(stock_data['Open'], stock_data['Close'])
        stock_data['Lower_Shadow'] = np.minimum(stock_data['Open'], stock_data['Close']) - stock_data['Low']
        
        # Price position within the day's range
        stock_data['Close_Position'] = (stock_data['Close'] - stock_data['Low']) / (stock_data['High'] - stock_data['Low'])
        
        # Remove rows with NaN (first and last rows)
        stock_data = stock_data.dropna()
        
        return stock_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def train_prediction_models(data):
    """Train models to predict next day's closing price"""
    
    # Features: Current day's OHLC + Previous close + Technical indicators
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Prev_Close',
        'Price_Range', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Close_Position'
    ]
    
    X = data[feature_columns]
    y = data['Next_Close']  # Target: Next day's closing price
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train models
    trained_models = {}
    model_performance = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        trained_models[name] = model
        model_performance[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MSE': mse
        }
    
    return trained_models, model_performance, feature_columns, X_test, y_test

# Load and prepare data
with st.spinner("🔄 Loading and preparing data..."):
    stock_data = load_and_prepare_data()

if stock_data is not None:
    # Train models
    with st.spinner("🤖 Training AI models..."):
        trained_models, model_performance, feature_columns, X_test, y_test = train_prediction_models(stock_data)
    
    # Sidebar - Model Performance
    st.sidebar.title("🏆 Model Performance")
    performance_df = pd.DataFrame(model_performance).T
    st.sidebar.dataframe(performance_df.round(4))
    
    best_model_name = performance_df['R2'].idxmax()
    st.sidebar.success(f"🥇 Best Model: {best_model_name}")
    st.sidebar.info(f"R² Score: {performance_df.loc[best_model_name, 'R2']:.4f}")
    
    # Main prediction interface
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("📊 Enter Today's Stock Data")
    
    with st.form("stock_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("📈 Price Data**")
            open_price = st.number_input("Open Price ($)", min_value=0.01, value=40.0, step=0.01, 
                                       help="Opening price of the stock today")
            high_price = st.number_input("High Price ($)", min_value=0.01, value=42.0, step=0.01,
                                       help="Highest price reached today")
        
        with col2:
            st.markdown("📉 Price Data**")
            low_price = st.number_input("Low Price ($)", min_value=0.01, value=38.0, step=0.01,
                                      help="Lowest price reached today")
            close_price = st.number_input("Close Price ($)", min_value=0.01, value=41.0, step=0.01,
                                        help="Current/closing price today")
        
        # Previous close (optional)
        prev_close = st.number_input("Previous Day's Close ($)", min_value=0.01, value=40.5, step=0.01,
                                   help="Yesterday's closing price (for better accuracy)")
        
        # Model selection
        model_choice = st.selectbox("🤖 Choose AI Model:", list(trained_models.keys()),
                                  help="Select which trained model to use for prediction")
        
        predict_button = st.form_submit_button("🔮 Predict Tomorrow's Price", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_button:
        # Validate inputs
        if high_price < max(open_price, close_price, low_price):
            st.error("❌ High price should be the highest value!")
        elif low_price > min(open_price, close_price, high_price):
            st.error("❌ Low price should be the lowest value!")
        else:
            # Calculate technical features
            price_range = high_price - low_price
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            close_position = (close_price - low_price) / price_range if price_range > 0 else 0.5
            
            # Prepare input data
            input_data = pd.DataFrame({
                'Open': [open_price],
                'High': [high_price],
                'Low': [low_price],
                'Close': [close_price],
                'Prev_Close': [prev_close],
                'Price_Range': [price_range],
                'Body_Size': [body_size],
                'Upper_Shadow': [upper_shadow],
                'Lower_Shadow': [lower_shadow],
                'Close_Position': [close_position]
            })
            
            # Make prediction
            selected_model = trained_models[model_choice]
            predicted_price = selected_model.predict(input_data)[0]
            
            # Calculate prediction confidence
            model_r2 = model_performance[model_choice]['R2']
            model_rmse = model_performance[model_choice]['RMSE']
            confidence_interval = 1.96 * model_rmse  # 95% confidence interval
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Tomorrow's Predicted Closing Price</h2>
                <h1>${predicted_price:.2f}</h1>
                <p><strong>Model Used:</strong> {model_choice}</p>
                <p><strong>Confidence Range:</strong> ${predicted_price - confidence_interval:.2f} - ${predicted_price + confidence_interval:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate insights
            price_change = predicted_price - close_price
            price_change_pct = (price_change / close_price) * 100
            
            # Display insights
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price Change", f"${price_change:.2f}", 
                         delta=f"{price_change_pct:.2f}%")
            
            with col2:
                direction = "📈 UP" if price_change > 0 else "📉 DOWN" if price_change < 0 else "➡ FLAT"
                st.metric("Direction", direction)
            
            with col3:
                confidence_pct = model_r2 * 100
                st.metric("Model Confidence", f"{confidence_pct:.1f}%")
            
            with col4:
                volatility = (price_range / open_price) * 100
                risk = "HIGH" if volatility > 5 else "MEDIUM" if volatility > 2 else "LOW"
                st.metric("Risk Level", risk)
            
        #  st.balloons()
            
            # Technical Analysis Summary
            st.subheader("📊 Technical Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("📈 Pattern Analysis:")
                if close_price > open_price:
                    candle_type = "🟢 Bullish (Green Candle)"
                elif close_price < open_price:
                    candle_type = "🔴 Bearish (Red Candle)"
                else:
                    candle_type = "➖ Doji (Neutral)"
                
                st.write(f"- Candle Pattern: {candle_type}")
                st.write(f"- Body Size: ${body_size:.2f} ({(body_size/price_range)*100:.1f}% of range)")
                st.write(f"- Upper Shadow: ${upper_shadow:.2f}")
                st.write(f"- Lower Shadow: ${lower_shadow:.2f}")
            
            with analysis_col2:
                st.markdown("🎯 Position Analysis:")
                if close_position > 0.7:
                    position_desc = "Near High (Strong)"
                elif close_position > 0.3:
                    position_desc = "Middle Range (Neutral)"
                else:
                    position_desc = "Near Low (Weak)"
                
                st.write(f"- Close Position: {close_position:.2f} ({position_desc})")
                st.write(f"- Daily Range: ${price_range:.2f}")
                st.write(f"- Volatility: {volatility:.2f}%")
    
    # # Historical Performance Chart
    # st.subheader("📈 Recent Price History")
    # recent_data = stock_data.tail(30)  # Last 30 days
    
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(recent_data['Date'], recent_data['Close'], marker='o', linewidth=2, markersize=4)
    # ax.set_title('Last 30 Days Stock Price')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Close Price ($)')
    # ax.grid(True, alpha=0.3)
    # plt.xticks(rotation=45)
    # st.pyplot(fig)
    
else:
    st.error("❌ Could not load stock data. Please ensure 'stock.csv' exists in the current directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🤖 <strong>AI Stock Price Predictor</strong> | Powered by Machine Learning</p>
    <p>⚠ <em>For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)