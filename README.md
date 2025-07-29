# 🚶‍♀️ Walkwise: An Agentic AI Coach for Your Walking Life

**Walkwise** is a data-powered Streamlit app that interprets your walking behaviour from Apple Watch data and turns it into personalized insights and motivational support. It's not just a dashboard — it's a digital companion that learns from your patterns and talks to you like a coach would.

This is an exploration of *agentic AI*, where systems aren’t passive tools but active, contextual collaborators. Meet your two key agents: **Stride** and **Karen**.

---

## 🔁 Overview

| Agent      | Type           | Core Function                                                                | AI-powered? |
|------------|----------------|-----------------------------------------------------------                   |-------------|
| **Stride** | Data Analyst   | Uses SARIMAX+Garch model to predict future steps and tailors messaging       | ❌          |
| **Karen**  | AI Coach       | clusters and Motivates and reflects on your walk using LLM reasoning         | ✅ GPT-4o        |

---

## 🧠 What Does It Do?

These clusters help you understand *how* you walk — not just *how much*. This agent sits quietly in the background, acting like a behavioral data analyst.

#### ✅ Tech Stack:
- Pandas, Scikit-learn
- K-Means clustering
- Apple Watch data (.csv)
- Matplotlib + Streamlit visualizations

Karen is an *agentic AI*. She doesn’t just display data — she speaks with empathy, encouragement, and contextual understanding. This isn’t a chatbot bolted on top of analytics. This is human-centered AI coaching.

#### ✅ Tech Stack:
- OpenAI GPT-4o (via API)
- Streamlit UI
- Dynamic prompt engineering
- Behaviour-aware conversation

---

## 📂 Folder Structure

walkwise-app/
├── app.py # Main app with Stride
├── pages/
│ └── karen.py # Separate page for GPT-powered agent Karen
├── data/
│ └── apple_watch_steps.csv # Sample or live step data
├── .env # Your OpenAI API key (not uploaded)
├── .gitignore
├── requirements.txt
└── README.md


---

## 🚀 How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Vaswati101/walkwise-app.git
   cd walkwise-app
Install dependencies:
pip install -r requirements.txt
Add your OpenAI key in .env:
OPENAI_API_KEY=your_key_here
Run the app:
streamlit run app.py
💼 Why This Matters for Employers

This project demonstrates:

🔬 Practical ML Use – Not toy models. Clustering applied to real-life behavioral data.
🤝 Human-Centered Design – Karen uses tone, context, and behavioral framing.
🤖 Agentic AI Principles – AI that acts and responds meaningfully to the user.
📈 Product Thinking – Visual storytelling, dashboards, and feedback loops built in.
🔧 End-to-End Execution – From data ingestion → clustering → AI feedback → UI.
🔐 Security Aware – Secrets handled via .env, with .gitignore best practices.
🛤️ Future Enhancements

Real-time sync with Apple Health or Google Fit
Additional agents (e.g., predictor, nudger, emotional companion)
Deployable via Streamlit Cloud, Hugging Face, or Docker
👤 Creator

Vaswati Hazarika
