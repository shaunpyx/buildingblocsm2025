import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    "Year": [1996, 2000, 2005, 2010, 2015, 2020, 2024],
    "Mammals": [1100, 1200, 1300, 1450, 1600, 1750, 1890],
    "Birds": [900, 950, 1020, 1150, 1300, 1425, 1550],
    "Reptiles": [450, 500, 600, 720, 850, 980, 1100],
    "Amphibians": [200, 280, 400, 540, 670, 810, 950],
    "Fishes": [1200, 1350, 1500, 1650, 1800, 1950, 2100],
    "Total": [4850, 5280, 5920, 6800, 7970, 8915, 9590]
}
df = pd.DataFrame(data)


models = {}
future_years = pd.DataFrame({"Year": [2025, 2026, 2027, 2028, 2029, 2030]})
predictions = {"Year": future_years["Year"].tolist()}

for category in df.columns[1:]:  
    X = df[["Year"]]
    y = df[category]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    models[category] = model
    predictions[category] = model.predict(future_years).astype(int).tolist()

future_df = pd.DataFrame(predictions)
st.title("Extinction Risk Explorer (1996-2030)")
st.write("Analyze extinction trends across different taxonomic groups.")
st.write("### 📊 Threatened Species Trends (1996-2024)")
fig, ax = plt.subplots()
for category in df.columns[1:-1]:
    ax.plot(df["Year"], df[category], marker="o", linestyle="-", label=category)
    for i, txt in enumerate(df[category]):
        ax.annotate(txt, (df["Year"].iloc[i], df[category].iloc[i]), textcoords="offset points", xytext=(0,5), ha='center')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Threatened Species")
ax.set_title("Threatened Species Trends by Category")
ax.legend()
ax.grid()
st.pyplot(fig)
st.write("### Predicted Extinction Risks (2025-2030)")
st.table(future_df)
tax_group = st.selectbox("Select a Species Group for Detailed Insights:", df.columns[1:])
st.write(f"### Detailed Trend Analysis: {tax_group}")
fig, ax = plt.subplots()
ax.plot(df["Year"], df[tax_group], marker="o", linestyle="-", label="Historical Data")
ax.plot(future_df["Year"], future_df[tax_group], marker="o", linestyle="--", color="red", label="Predicted Data")
for i, txt in enumerate(df[tax_group]):
    ax.annotate(txt, (df["Year"].iloc[i], df[tax_group].iloc[i]), textcoords="offset points", xytext=(0,5), ha='center')
for i, txt in enumerate(future_df[tax_group]):
    ax.annotate(txt, (future_df["Year"].iloc[i], future_df[tax_group].iloc[i]), textcoords="offset points", xytext=(0,5), ha='center', color='red')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Threatened Species")
ax.set_title(f"Trend of {tax_group} Threatened Species")
ax.legend()
ax.grid()
st.pyplot(fig)
