import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import random


encouraging_msgs = [
    "Amazing job! You crushed your goal by {diff} steps! 🎉 Keep it up!",
    "You walked {steps} steps yesterday—way above the forecast! 🚀",
    "Fantastic effort! {diff} steps more than expected! You're unstoppable! 💪",
    "You're smashing those goals! {steps} steps is impressive! Keep shining! ✨",
    "Great hustle! Walking {diff} extra steps means you’re on fire! 🔥",
    "You exceeded predictions by {diff} steps yesterday — awesome work! 🎯",
    "Way to go! Your {steps} steps blew past expectations! 🏅",
    "What a power walk! {diff} steps beyond forecast, keep moving! 👣",
    "Superb effort! You've beaten the predicted steps by {diff}! 🥳",
    "You’re a step champ! {steps} steps is no joke — keep it rolling! 🚶‍♀️",
    "Killer performance with {steps} steps, way above the prediction! 💥",
    "You've gone the extra mile—actually, many miles! {diff} steps more! 🏃‍♂️",
    "Stellar walk yesterday with {steps} steps! Keep that energy high! ⚡",
    "Crushed it! {diff} steps beyond forecast means you're on fire! 🔥",
    "You're walking on sunshine with {steps} steps! Keep glowing! ☀️",
    "More steps, more wins! You outdid yourself by {diff} steps! 🥇",
    "Your feet must be smiling with that {steps} step count! 😄",
    "You went above and beyond with {diff} extra steps! Proud of you! 🙌",
    "Every step counts, and you nailed {steps} yesterday! Keep it up! 💪",
    "Outstanding work walking {steps} steps—you're killing it! 🎯",
    "Your steps were on fire yesterday—{diff} steps ahead! 🔥🔥",
    "Awesome strides made with {steps} steps—way to go! 🚀",
    "You beat the forecast and then some—{diff} steps more! 🎉",
    "Keep that pace! {steps} steps is seriously impressive! 👏",
    "You're setting the bar high with {diff} steps beyond prediction! 🏆",
    "Bravo! {steps} steps is a great achievement! Keep shining! ✨",
    "Your step count is soaring—{diff} steps more than forecast! 🌟",
    "You're outpacing expectations with {steps} steps! Go you! 💨",
    "Exceptional day walking {steps} steps, keep it rolling! 🚶‍♂️",
    "You're walking like a champion with {diff} steps beyond forecast! 🥇",
    "Your energy is contagious! {steps} steps and counting! 🔋",
    "You blew past the prediction—{diff} extra steps! Incredible! 💥",
    "Amazing strides yesterday—{steps} steps and climbing! 🌄",
    "Your feet are on fire! {diff} steps more than expected! 🔥",
    "You're smashing goals with {steps} steps—keep that fire alive! 🔥",
    "Your walk was legendary—{diff} steps beyond prediction! 🏅",
    "Incredible effort with {steps} steps—keep the momentum! 🚀",
    "You’re unstoppable with {diff} extra steps yesterday! 💪",
    "Great job outwalking expectations—{steps} steps strong! 👣",
    "Your steps are breaking records—{diff} steps above forecast! 🥳",
    "A phenomenal day of walking with {steps} steps! Keep shining! ✨",
    "Way to outpace the forecast by {diff} steps! You're on fire! 🔥",
    "You’re walking proof that hard work pays off—{steps} steps! 👏",
    "Incredible energy shown with {diff} steps beyond forecast! 💥",
    "Your step count was off the charts—{steps} steps! 🏆",
    "You hit a home run with {diff} extra steps yesterday! ⚾",
    "You're a walking superstar with {steps} steps! 🌟",
    "You've outperformed with {diff} extra steps! Keep it up! 🙌",
    "Your walk was a masterpiece—{steps} steps and counting! 🎨",
    "You crushed it with {diff} steps above forecast! 🚀",
]


gentle_msgs = [
    "A nice balanced day with {steps} steps — hope you’re feeling good! 😊",
    "Steady progress: {steps} steps yesterday, keep the rhythm going! 🎵",
    "A calm day with {steps} steps is just what you need sometimes. 🌿",
    "Good work! {steps} steps means you’re pacing yourself well. 🧘‍♂️",
    "Yesterday was a medium effort day — rest well and recharge! 🔋",
    "Moderate steps at {steps}, just the right pace for today. 👍",
    "A gentle stroll of {steps} steps — hope you enjoyed it! 🌸",
    "Taking it easy with {steps} steps — perfect balance! ⚖️",
    "Nice job on your walk yesterday, {steps} steps is solid! 🚶‍♂️",
    "A steady {steps} steps — consistency is key! 🔑",
    "You’re keeping a smooth pace with {steps} steps! Keep it steady! 🛤️",
    "Yesterday’s {steps} steps were just right for a good day. 🌞",
    "Solid effort with {steps} steps — a balanced approach! ⚖️",
    "A calm walk with {steps} steps — sometimes less is more! 🌼",
    "A good day with {steps} steps, keep listening to your body! 💆‍♀️",
    "Comfortable pace at {steps} steps — great job maintaining balance! 🧘‍♀️",
    "A moderate walk of {steps} steps, perfect for recovery! 🌱",
    "Your {steps} steps yesterday set a nice steady tone! 🎶",
    "Steady as she goes! {steps} steps is a solid pace. 🚤",
    "Just the right amount of movement — {steps} steps yesterday. 😊",
    "You’re on a smooth path with {steps} steps! Keep it up! 🛤️",
    "Your steps were moderate at {steps} — rest and recovery count too! 🌙",
    "Great job pacing yourself with {steps} steps. Balance is key! ⚖️",
    "A gentle day with {steps} steps — recharge and relax! 🛌",
    "Nice steady effort with {steps} steps — consistency wins! 🏆",
    "Yesterday’s {steps} steps shows good balance and care! 🧡",
    "A comfortable {steps} steps — perfect for a steady day! 🐢",
    "You’re pacing well with {steps} steps — keep it consistent! 🏃‍♂️",
    "Your {steps} steps are just right for today’s energy. 🌟",
    "Easy does it — {steps} steps yesterday, good job! ✨",
    "A steady {steps} steps — your consistency is impressive! 🔒",
    "You maintained a good flow with {steps} steps yesterday! 🌊",
    "Balanced steps at {steps} — well done pacing yourself! 🧘",
    "A nice calm walk of {steps} steps — balance is everything! ⚖️",
    "Good pace with {steps} steps — don’t forget to rest too! 🛀",
    "Your steps were just right yesterday — {steps} is solid! 💪",
    "Moderate walking with {steps} steps — listening to your body! 👂",
    "A calm and steady {steps} steps — great approach! 🌿",
    "You’ve found a nice rhythm with {steps} steps. Keep it up! 🎵",
    "Steady and strong with {steps} steps — great consistency! 🦾",
    "Yesterday’s {steps} steps was a nice, balanced effort! 🌸",
    "Well-paced with {steps} steps — keep that rhythm going! 🎼",
    "Consistent steps at {steps} — you’re on a great track! 🚂",
    "A nice steady walk with {steps} steps — keep the balance! ⚖️",
    "Good job pacing yourself at {steps} steps — consistency is key! 🔑",
]


reminder_msgs = [
    "Looks like no steps were recorded yesterday — hope you’re okay! 💙",
    "No step data found yesterday. Maybe a rest day or forgot your watch? ⌚",
    "No steps logged — don’t forget to wear your watch today! 🕒",
    "No steps detected yesterday. Hope all is well with you! 🤗",
    "Missing step data — a gentle reminder to keep moving! 🏃‍♀️",
    "Oops! No recorded steps. Did you have a break or is your watch off? 🤔",
    "Step count was zero — remember to keep active and hydrated! 💧",
    "No steps logged yesterday. Take it slow and steady today! 🐢",
    "Step data missing — a perfect day to start fresh! 🌞",
    "No steps recorded — your watch might need a little charge! 🔋",
    "Seems like yesterday was a rest day — hope you’re recharging! 🌙",
    "No steps found — maybe your watch took a nap too? 😴",
    "No step data — a little nudge to get moving today! 🚶‍♂️",
    "No recorded steps — hope everything’s alright with you! 🙏",
    "Step count missing — don’t forget to wear your tracker! ⌚",
    "No steps logged yesterday — hope you had a relaxing day! 🛌",
    "No steps recorded — maybe a good day for some light stretching? 🤸‍♀️",
    "No step data found — your watch might be feeling lazy! 😅",
    "No steps yesterday — time to get those feet moving again! 🦶",
    "No recorded steps — remember to stay hydrated and active! 💧",
    "Step count is zero — hope you’re taking good care of yourself! 💙",
    "No steps logged — maybe time for a gentle walk today? 🌼",
    "Missing steps — your watch might be charging up! 🔌",
    "No step data — don’t forget to keep your tracker close! 📱",
    "No recorded steps yesterday — hope you’re well rested! 🌟",
    "Step count zero — take it easy and listen to your body! 🧘‍♂️",
    "No steps found — your watch might be on vacation too! 🏖️",
    "No steps logged — maybe time for a quick stroll? 🚶‍♀️",
    "No recorded steps — hope all is good on your end! 🤗",
    "No step data — a gentle reminder to stay active! 🏃‍♂️",
    "No steps logged yesterday — hope your watch is charged! ⚡",
    "No steps recorded — remember movement is medicine! 💊",
    "No recorded steps — a calm day can be just as good! 🌿",
    "No steps found — hope you’re taking care of yourself! 💚",
    "No step data — your watch might be feeling shy today! 🤭",
    "No steps logged — maybe a good day for some yoga? 🧘‍♀️",
    "No steps recorded — hope you’re enjoying some rest! 😌",
    "Step count zero — every day can’t be a marathon! 🐢",
    "No steps found — remember to take breaks and breathe! 🌬️",
    "No recorded steps — hope your watch is still on your wrist! 😉",
    "No step data — a little nudge to get moving gently today! 🦶",
    "No steps logged — hope you had a peaceful day! 🌙",
    "No steps recorded — a reminder that rest is important too! 🌸",
    "No recorded steps — your body knows best! Listen to it! 💖",
    "No step data found — hope you’re feeling good! 😊",
    "No steps logged — maybe a quiet day was needed! 🌾",
    "No steps recorded — rest well and recharge for tomorrow! 🔋",
    "No steps logged — hope your watch didn't take a break! 😄",
    "No steps found — looking forward to your next walk! 🚶‍♂️",
]


def get_stride_message(steps, predicted_steps):
    diff = steps - predicted_steps
    if steps == 0:
        return random.choice(reminder_msgs)
    elif diff > 500:  # Threshold can be adjusted
        return random.choice(encouraging_msgs).format(steps=steps, diff=diff)
    else:
        return random.choice(gentle_msgs).format(steps=steps)

st.title("Stride: Steps Forecast & Feedback")

@st.cache_data
def load_data():
    df = pd.read_csv('/Users/vaswatihazarika/Desktop/walkmate/one/data/apple_watch_steps.csv',  parse_dates=['date'], index_col='date')
    df = df.sort_index()
    all_days = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(all_days, fill_value=0)
    df.index.name = 'date'
    return df

df = load_data()

# === Section 1: Filter and View ===
st.header("#1 Filter and View")

start_date = st.date_input("Start Date", df.index.min().date())
end_date = st.date_input("End Date", df.index.max().date())
min_steps = st.number_input("Minimum Steps", min_value=0, value=0, step=1000)

def filter_data(df, start_date, end_date, min_steps):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    filtered = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    filtered = filtered[filtered['steps'] >= min_steps]
    return filtered

filtered_df = filter_data(df, start_date, end_date, min_steps)

st.subheader("Filtered Data Preview")
st.write(filtered_df)

# === Section 2: Fit and Forecast (Box-Cox + GARCH) ===
st.header("#2 Fit and Forecast (with Box-Cox + GARCH)")

if len(filtered_df) < 30:
    st.warning("Not enough data points (need at least 30) to fit the model.")
else:
    if st.button("Fit Advanced Model & Forecast Next 5 Days"):
        with st.spinner("Fitting Box-Cox SARIMAX and GARCH model..."):
            # Box-Cox transform (add 1 to avoid zeros)
            steps_adj = filtered_df['steps'] + 1
            steps_boxcox, fitted_lambda = boxcox(steps_adj)

            # Fit SARIMAX on Box-Cox transformed data
            model = SARIMAX(steps_boxcox, order=(2,1,2))
            results = model.fit(disp=False)

            # Fit GARCH(1,2) on residuals
            residuals = results.resid
            garch_model = arch_model(residuals, vol='GARCH', p=1, q=2)
            garch_results = garch_model.fit(disp="off")

            # Forecast ARIMA mean for next 5 days
            arima_forecast = results.get_forecast(steps=5)
            forecast_index = pd.date_range(filtered_df.index[-1] + pd.Timedelta(days=1), periods=5)
            mean_forecast = arima_forecast.predicted_mean

            # Forecast GARCH variance
            garch_forecast = garch_results.forecast(horizon=5)
            variance_forecast = garch_forecast.variance.values[-1, :]
            std_forecast = np.sqrt(variance_forecast)

            # 95% confidence intervals on Box-Cox scale
            z = 1.96
            lower_bc = mean_forecast - z * std_forecast
            upper_bc = mean_forecast + z * std_forecast

            # Inverse Box-Cox transform to original scale
            mean_forecast_orig = inv_boxcox(mean_forecast, fitted_lambda)
            lower_orig = inv_boxcox(lower_bc, fitted_lambda)
            upper_orig = inv_boxcox(upper_bc, fitted_lambda)

        st.subheader("Forecast for Next 5 Days")
        forecast_df = pd.DataFrame({
            'forecast': mean_forecast_orig,
            'lower_ci': lower_orig,
            'upper_ci': upper_orig
        }, index=forecast_index)
        st.write(forecast_df)

        # Plot historical + forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_df.index, filtered_df['steps'], label="Historical Steps")
        ax.plot(forecast_df.index, forecast_df['forecast'], color='red', label="Forecast")
        ax.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'], color='pink', alpha=0.3, label="95% CI")
        ax.set_ylabel("Steps")
        ax.legend()
        st.pyplot(fig)

        # Save results for Section 3 (feedback)
        st.session_state['forecast_df'] = forecast_df
        st.session_state['filtered_df'] = filtered_df

# === Section 3: Stride Says ===
st.header("#3 Stride Says")

if 'forecast_df' in st.session_state and 'filtered_df' in st.session_state:
    last_actual = st.session_state['filtered_df']['steps'][-1]
    last_forecast = st.session_state['forecast_df']['forecast'][0]

    message = get_stride_message(last_actual, last_forecast)
    st.markdown(f"### {message}")

    # Optional: show actual vs predicted for clarity
    st.write(f"**Actual Steps:** {last_actual}")
    st.write(f"**Predicted Steps:** {int(last_forecast)}")

else:
    st.write("Fit and forecast some data to see feedback here.")

