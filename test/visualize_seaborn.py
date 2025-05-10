"""
visualize_seaborn.py

This script demonstrates how to load the Erbil real estate data and create
basic data visualizations using Seaborn and Matplotlib.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load the dataset
    df = pd.read_excel("erbil_data.xlsx", engine="openpyxl")

    # 2. Clean and standardize column names (similar to your train_model script)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\n", " ", regex=False)

    # 3. Rename the target column for convenience
    target_col = "Sale price of the property in U$"
    # Drop rows missing the target
    df.dropna(subset=[target_col], inplace=True)

    # Convert the target to numeric (if needed)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df.dropna(subset=[target_col], inplace=True)

    # 4. Example visualization: Distribution of Sale Price
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=target_col, kde=True, color="blue", bins=30)
    plt.title("Distribution of Sale Price")
    plt.xlabel("Sale Price (USD)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("sale_price_distribution.png", dpi=100)
    plt.show()

    # 5. Boxplot of Sale Price by Zone/Location (if that column exists)
    location_col = "Zone/Location"
    if location_col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=location_col, y=target_col, data=df)
        plt.title("Sale Price by Zone/Location")
        plt.xlabel("Zone/Location")
        plt.ylabel("Sale Price (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("sale_price_by_location.png", dpi=100)
        plt.show()

    # 6. Correlation Heatmap (for numeric columns only)
    #    Identify numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=100)
    plt.show()

    print("Seaborn visualizations saved successfully.")

if __name__ == "__main__":
    main()
