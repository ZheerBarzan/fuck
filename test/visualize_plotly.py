"""
visualize_plotly.py

Demonstrates how to load the Erbil real estate data and create
interactive data visualizations using Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    # 1. Load the dataset
    df = pd.read_excel("erbil_data.xlsx", engine="openpyxl")

    # 2. Clean up column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\n", " ", regex=False)

    # 3. Target column
    target_col = "Sale price of the property in U$"
    df.dropna(subset=[target_col], inplace=True)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df.dropna(subset=[target_col], inplace=True)

    # 4. Interactive Histogram of Sale Price
    fig_hist = px.histogram(
        df,
        x=target_col,
        nbins=30,
        title="Distribution of Sale Price (Interactive)",
        labels={target_col: "Sale Price (USD)"}
    )
    fig_hist.show()

    # 5. Interactive Boxplot of Sale Price by Zone/Location
    location_col = "Zone/Location"
    if location_col in df.columns:
        fig_box = px.box(
            df,
            x=location_col,
            y=target_col,
            title="Sale Price by Zone/Location (Interactive)",
            labels={
                location_col: "Zone/Location",
                target_col: "Sale Price (USD)"
            }
        )
        fig_box.update_layout(xaxis={'categoryorder':'total descending'})
        fig_box.show()

    # 6. Interactive Correlation Heatmap
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr_matrix = df[numeric_cols].corr().round(2)

    # Convert correlation matrix to a "long" format suitable for heatmap
    corr_long = corr_matrix.stack().reset_index()
    corr_long.columns = ['Variable1', 'Variable2', 'Correlation']

    # Build the heatmap
    fig_heatmap = px.imshow(
        corr_matrix,
        x=numeric_cols,
        y=numeric_cols,
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap (Interactive)",
        text_auto=True,
        aspect="auto"
    )
    fig_heatmap.show()

    print("Plotly interactive visualizations displayed successfully.")

if __name__ == "__main__":
    main()
