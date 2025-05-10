"""
advisor.py

A Streamlit app that suggests the top 3 investment locations in Erbil based on user's budget.
Analyzes historical data to provide investment recommendations with rationales.

⭕ streamlit run advisor.py
Model-based recommendations unavailable. Using historical scoring.
  Network URL: http://100.101.15.36:8501

Transformation Pipeline and Model Successfully Loaded
[Model Recommendation Error] 375 missing values found in the target column: Sale price of the property in U$. To proceed, remove the respective rows from the data.



"""

import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model

@st.cache_resource
def get_model():
    return load_model("best_price_model")

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the Erbil real estate data for analysis."""
    try:
        df = pd.read_excel("erbil_data.xlsx", engine="openpyxl")
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace("\n", " ", regex=False)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def analyze_location_metrics(df, budget):
    """
    Analyze locations based on comprehensive property metrics
    """
    location_metrics = []
    
    for location in df["Zone/Location"].unique():
        location_data = df[df["Zone/Location"] == location]
        
        # Skip if insufficient data
        if len(location_data) < 3:
            continue
            
        # Basic metrics
        avg_price = location_data["Sale price of the property in U$"].mean()
        avg_price_sqm = (location_data["Sale price of the property in U$"] / 
                        location_data["Land area"]).mean()
        
        # Feature-based scoring
        feature_score = 0
        if "Number of bedrooms" in location_data.columns:
            feature_score += location_data["Number of bedrooms"].mean() * 0.1
        if "Number of bathrooms" in location_data.columns:
            feature_score += location_data["Number of bathrooms"].mean() * 0.1
        
        # Location quality score based on additional features
        location_quality = 0
        features_to_check = ["Proximity to amenities", "Parking availability", 
                           "Garden or outdoor space", "Heating/cooling systems"]
        for feature in features_to_check:
            if feature in location_data.columns:
                feature_present = location_data[feature].notna().mean()
                location_quality += feature_present * 0.25
        
        # Price range and budget analysis
        price_range = location_data["Sale price of the property in U$"].std()
        budget_fit = 1 - min(abs(avg_price - budget) / budget, 1)
        
        # Recent market activity
        current_year = df["Year"].max()
        recent_transactions = len(location_data[location_data["Year"] >= current_year - 1])
        transaction_score = min(recent_transactions / 10, 1)  # Normalize to 0-1
        
        # Property age consideration
        avg_age = location_data["Age of the property in years"].mean()
        age_score = 1 - min(avg_age / 10, 1)  # Newer properties score higher
        
        # Calculate growth rate with year-over-year comparison
        yearly_prices = location_data.groupby("Year")["Sale price of the property in U$"].mean()
        if len(yearly_prices) >= 2:
            growth_rate = (yearly_prices.iloc[-1] / yearly_prices.iloc[0] - 1)
            growth_rate = min(max(growth_rate, 0), 1)  # Normalize between 0 and 1
        else:
            growth_rate = 0
        
        # Updated scoring weights
        investment_score = (
            budget_fit * 0.25 +          # Budget match
            growth_rate * 0.15 +         # Price appreciation
            transaction_score * 0.15 +    # Market activity
            feature_score * 0.15 +        # Property features
            location_quality * 0.15 +     # Location amenities
            age_score * 0.15             # Property age
        )
        
        location_metrics.append({
            "location": location,
            "avg_price": avg_price,
            "avg_price_sqm": avg_price_sqm,
            "price_range": price_range,
            "recent_transactions": recent_transactions,
            "growth_rate": growth_rate,
            "investment_score": investment_score,
            "budget_fit": budget_fit,
            "feature_score": feature_score,
            "location_quality": location_quality,
            "avg_age": avg_age
        })
    
    return sorted(location_metrics, key=lambda x: x["investment_score"], reverse=True)

def generate_location_insights(location_data, budget):
    """Generate detailed insights for a location recommendation."""
    insights = []
    
    # Price analysis with more detail
    if location_data["avg_price"] < budget * 0.8:
        insights.append("Significantly below budget - excellent value potential")
    elif location_data["avg_price"] < budget:
        insights.append("Within budget - good value proposition")
    elif location_data["avg_price"] <= budget * 1.2:
        insights.append("Slightly above budget - consider negotiation")
    else:
        insights.append("Premium location - consider smaller properties")
    
    # Feature quality insights
    if location_data["feature_score"] > 0.7:
        insights.append("High-quality properties with excellent amenities")
    elif location_data["feature_score"] > 0.4:
        insights.append("Good property features and amenities")
    
    # Location quality insights
    if location_data["location_quality"] > 0.7:
        insights.append("Prime location with excellent accessibility")
    elif location_data["location_quality"] > 0.4:
        insights.append("Good location with essential amenities")
    
    # Age consideration
    if location_data["avg_age"] < 2:
        insights.append("Newer properties - minimal renovation needed")
    elif location_data["avg_age"] < 5:
        insights.append("Relatively new properties - good condition")
    else:
        insights.append("Consider potential renovation needs")
        
    # Market activity
    if location_data["recent_transactions"] > 5:
        insights.append("Active market with good liquidity")
    else:
        insights.append("Less liquid market - may offer negotiation opportunities")
        insights.append("Potential value opportunity")

    return insights

import asyncio

async def async_operation():
    # Your async code here
    pass

# In your main code
if __name__ == "__main__":
    asyncio.run(async_operation())

def get_model_recommendations(df, budget):
    try:
        model = get_model()
        df_pred = df.copy()
        df_pred = df_pred[df_pred["Sale price of the property in U$"].notna()]
        if df_pred.empty:
            return []
        preds = predict_model(model, data=df_pred)
        pred_col = None
        for col in ["prediction_label", "Label", "Score", "Predicted_Price"]:
            if col in preds.columns:
                pred_col = col
                break
        if pred_col is None:
            raise KeyError(f"Prediction column not found in predict_model output: {list(preds.columns)}")
        df_pred["Predicted_Price"] = preds[pred_col]
        # For each location, find the best property for the budget
        locations = []
        for location in df_pred["Zone/Location"].unique():
            loc_data = df_pred[df_pred["Zone/Location"] == location]
            # Only consider properties within budget
            in_budget = loc_data[loc_data["Predicted_Price"] <= budget]
            if in_budget.empty:
                continue
            # Best fit: property with highest predicted price <= budget
            best_prop = in_budget.loc[in_budget["Predicted_Price"].idxmax()]
            # Growth rate (historical)
            if "Year" in loc_data.columns:
                yearly_prices = loc_data.groupby("Year")["Sale price of the property in U$"].mean()
                if len(yearly_prices) >= 2:
                    growth_rate = (yearly_prices.iloc[-1] / yearly_prices.iloc[0] - 1)
                else:
                    growth_rate = 0
            else:
                growth_rate = 0
            # Liquidity: recent transactions (last 1 year)
            if "Year" in loc_data.columns:
                current_year = loc_data["Year"].max()
                recent_transactions = len(loc_data[loc_data["Year"] >= current_year - 1])
            else:
                recent_transactions = np.nan
            # Price per m²
            if best_prop["Land area"] > 0:
                price_per_m2 = best_prop["Predicted_Price"] / best_prop["Land area"]
            else:
                price_per_m2 = np.nan
            # Rationale
            rationale = []
            if abs(best_prop["Predicted_Price"] - budget) < budget * 0.1:
                rationale.append("Excellent budget fit")
            if growth_rate > 0.05:
                rationale.append(f"High growth rate ({growth_rate*100:.1f}% in recent years)")
            if growth_rate < 0:
                rationale.append("Warning: Negative growth rate (declining prices in recent years)")
            if recent_transactions > 5:
                rationale.append("Active market (high liquidity)")
            if price_per_m2 < 1000:
                rationale.append("Good value per m²")
            if best_prop["Age of the property in years"] < 5:
                rationale.append("Relatively new properties")
            if not rationale:
                rationale.append("Solid investment potential")
            locations.append({
                "location": location,
                "best_predicted_price": best_prop["Predicted_Price"],
                "actual_price": best_prop["Sale price of the property in U$"],
                "land_area": best_prop["Land area"],
                "category": best_prop["Category"],
                "age": best_prop["Age of the property in years"],
                "growth_rate": growth_rate,
                "recent_transactions": recent_transactions,
                "price_per_m2": price_per_m2,
                "rationale": rationale,
                "sample": f"{best_prop['Category']} | {best_prop['Land area']}m² | ${best_prop['Predicted_Price']:,.0f} (actual: ${best_prop['Sale price of the property in U$']:,.0f})"
            })
        # Rank by best_predicted_price (closest to budget), then growth, then liquidity
        def sort_key(x):
            within_budget = float(x["best_predicted_price"] <= budget)
            return (-within_budget, -x["best_predicted_price"], -x["growth_rate"], -x["recent_transactions"])
        locations = sorted(locations, key=sort_key)
        return locations[:3]
    except Exception as e:
        print(f"[Model Recommendation Error] {e}")
        return None

def main():
    st.set_page_config(page_title="Erbil Investment Advisor", layout="wide")
    st.title("Erbil Real Estate Investment Advisor")
    st.write("""
    Welcome to the Erbil Investment Advisor! This tool analyzes real estate data to suggest 
    the best locations for your investment based on your budget and market conditions.
    """)
    df = load_and_prepare_data()
    if df is None:
        return
    st.subheader("What's your investment budget?")
    budget = st.number_input(
        "Investment Budget (USD)",
        min_value=10000,
        max_value=10000000,
        value=200000,
        step=10000,
        help="Enter your maximum investment budget in USD"
    )
    if st.button("Suggest Me Best Investment Locations"):
        with st.spinner("Analyzing market data using trained model..."):
            recommendations = get_model_recommendations(df, budget)
            if recommendations is None:
                st.error("Model-based recommendations unavailable. Please check your data and model.")
                return
            if not recommendations:
                st.error("No suitable investment locations found for your budget.")
                return
            st.subheader("Top 3 Model-Driven Investment Locations")
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    st.markdown(f"### #{i}: {rec['location']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Model Metrics:**")
                        st.write(f"- Best Fit Price: ${rec['best_predicted_price']:,.0f}")
                        st.write(f"- Actual Price: ${rec['actual_price']:,.0f}")
                        st.write(f"- Land Area: {rec['land_area']:,.0f} m²")
                        st.write(f"- Age: {rec['age']} years")
                        growth_str = f"{rec['growth_rate']*100:.1f}%"
                        if rec['growth_rate'] < 0:
                            st.write(f"- Growth Rate: :red[{growth_str}] (declining)")
                        else:
                            st.write(f"- Growth Rate: {growth_str}")
                        st.write(f"- Recent Transactions: {rec['recent_transactions']}")
                        st.write(f"- Price per m²: ${rec['price_per_m2']:,.0f}")
                    with col2:
                        st.write("**Why invest here?**")
                        for reason in rec['rationale']:
                            st.write(f"- {reason}")
                        st.write("**Sample Property:**")
                        st.write(f"- {rec['sample']}")
                    st.markdown("---")

if __name__ == "__main__":
    main()