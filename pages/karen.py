#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 20:30:03 2025

@author: vaswatihazarika
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 20:30:03 2025

@author: vaswatihazarika
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()


# === CLUSTERING FUNCTION ===
@st.cache_data
def load_and_cluster_data():
    df = pd.read_csv('data/apple_watch_steps.csv', parse_dates=['date'], index_col='date')
    df = df.sort_index()

    # Fill missing dates
    all_days = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(all_days, fill_value=0)
    df.rename_axis('date', inplace=True)

    # Scale steps
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['steps']])

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, max_iter=100, random_state=42)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)

    # Assign human-friendly cluster names based on average steps
    avg_steps = df.groupby('cluster_id')['steps'].mean()
    sorted_clusters = avg_steps.sort_values().index.tolist()
    name_map = {
        0: "Sedentary",
        1: "Moderately Active",
        2: "Highly Active",
        3: "Super Active"
    }
    cluster_name_map = {cid: name_map[i] for i, cid in enumerate(sorted_clusters)}
    df['cluster_name'] = df['cluster_id'].map(cluster_name_map)

    return df

# === Load data ===
df = load_and_cluster_data()

# === OpenAI GPT setup ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# === Streamlit Layout ===
st.title("👟 Meet Karen: Your Walking Buddy AI")

# 1. Cluster Overview
st.header("1. Your Walking Clusters Overview")
cluster_summary = df.groupby('cluster_name')['steps'].describe()[['mean', 'min', 'max']].round(0)
st.dataframe(cluster_summary)

# 2. Pie Chart
st.subheader("Distribution of Days by Cluster")
cluster_counts = df['cluster_name'].value_counts()
fig, ax = plt.subplots()
ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# 3. Latest Day
st.header("2. Your Most Recent Walking Day")
latest_date = df.index.max()
latest_day = df.loc[latest_date]
steps = latest_day['steps']
cluster_name = latest_day['cluster_name']

st.markdown(f"📅 **Date:** {latest_date.date()}")
st.markdown(f"👣 **Steps:** {steps}")
st.markdown(f"🔍 **Cluster:** `{cluster_name}`")

# 4. GPT Feedback
st.subheader("3. Karen's Encouragement (AI-powered)")

def karen_llm_feedback(cluster_name, steps):
    prompt = f"""
You are Karen, a friendly and supportive AI walking coach.
The user walked {steps} steps yesterday and falls into the "{cluster_name}" cluster.
Your job is to give them a short motivational message using a growth mindset tone.
End with a supportive suggestion if helpful.
Keep it kind, human-like and encouraging.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

if st.button("Get Karen's Feedback"):
    with st.spinner("Karen is thinking..."):
        message = karen_llm_feedback(cluster_name, steps)
    st.success(message)

# 5. GPT Q&A
st.header("4. Ask Karen Anything")
user_q = st.text_input("Ask Karen a question about walking, motivation, or fitness:")

if user_q:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Karen, a motivational walking coach AI. Keep it upbeat and growth-minded."},
            {"role": "user", "content": user_q}
        ]
    )
    st.markdown(f"**Karen says:** {response.choices[0].message.content.strip()}")
