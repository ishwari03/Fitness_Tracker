import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import plotly.graph_objects as go
import datetime
import calendar  # For the goal completion calendar
import warnings
warnings.filterwarnings('ignore')

if st.sidebar.selectbox("Select Color Theme:", ["Light Mode", "Dark Mode"]) == "Dark Mode":
    st.markdown(
        """
        <style>
        body {
            background-color: #1a1a1a;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #1a1a1a;
            color: #f0f0f0;
        }
        .stSlider > div > div > div > div {
            background-color: #0a0a0a;
        }
        h1, h2, h3, h4, h5, h6 { /* Target header elements */
            color: #ff5500; /* Orange color */
        }
         [data-testid="stSidebar"] {
            background-color: #2c2c2c; /* Dark sidebar background */
            color: #f0f0f0;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio > label {
            color: #f0f0f0;
        }
        div.stButton > button {
            background-color: black;
            color: white;
            border: none;
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f8ff;
        }
        .stSlider > div > div > div > div {
            background-color: #0a0a0a;
        }
         [data-testid="stSidebar"] {
            background-color: #e0f2f7; /* Light sidebar background */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if "exercise_df" not in st.session_state:
    st.session_state.exercise_df = None
if "df_user_input" not in st.session_state:
    st.session_state.df_user_input = None
if "total_calories_burned" not in st.session_state:
    st.session_state.total_calories_burned = 0
if "calorie_goal" not in st.session_state:
    st.session_state.calorie_goal = 100 # example calorie goal

def create_progress_gauge(progress):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=progress,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "green"}}
    ))
    return fig

# Placeholder for AI-based suggestions (replace with your model)
def generate_ai_suggestions(age, duration, gender, exercise_type, goal):
    # Workout Recommendations
    warmup = "Start with a 5-minute light cardio warm-up."
    cooldown = "Finish with a 5-minute stretching cool-down."
    if exercise_type == "Running":
        workout = f"For your {goal} goal, try a {duration + 10}-minute run at a comfortable pace. Focus on maintaining a steady rhythm."
    elif exercise_type == "Weightlifting":
        workout = f"Focus on compound exercises like squats, deadlifts, and bench presses. Perform 3 sets of 8-12 repetitions with proper form."
    elif exercise_type == "Swimming":
        workout = f"Try swimming for {duration + 10} minutes. Vary your strokes to work different muscle groups. Focus on proper breathing technique."
    elif exercise_type == "Cycling":
        workout = f"Try cycling for {duration + 10} minutes. Vary your terrain to work different muscle groups. Focus on maintaining a steady cadence."
    elif exercise_type == "Yoga":
        workout = f"Try a yoga routine for {duration + 10} minutes. Focus on controlled breathing and proper form. Try a routine focused on your fitness goal."
    else:
        workout = f"Try {exercise_type} for {duration + 10} minutes. Maintain a consistent pace and focus on proper form."

    # Diet Suggestions
    if goal == "Weight Loss":
        diet = "Aim for a calorie deficit of 500 calories per day. Focus on lean proteins, vegetables, and whole grains. Drink plenty of water."
    elif goal == "Muscle Gain":
        diet = "Increase your protein intake to 1.6-2.2 grams per kilogram of body weight. Include complex carbohydrates and healthy fats. Stay hydrated."
    elif goal == "Endurance":
        diet = "Focus on balanced nutrition with a mix of carbohydrates, protein, and healthy fats. Include fruits and vegetables for vitamins and minerals. Stay well-hydrated before, during, and after workouts."
    else:
        diet = "Maintain a balanced diet with a focus on nutrient-dense foods. Ensure adequate hydration and consume a variety of fruits and vegetables."

    # Stress & Recovery Insights
    recovery = "Prioritize sleep and rest days. Consider foam rolling, light stretching, or a warm bath. Practice mindfulness to reduce stress."

    # Sleep Quality Analysis
    sleep = "Establish a consistent sleep schedule. Avoid caffeine before bed."

    return [
        f"## Workout Recommendation:\n {warmup} {workout} {cooldown}",
        f"## Personalized Diet Suggestion:\n {diet}",
        f"## Stress & Recovery Insight:\n {recovery}",
        f"## Sleep Quality Analysis:\n {sleep}",
    ]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Dashboard", "Calorie Prediction", "AI Suggestions", "Progress Tracking","BMI Calculator"))

if page == "Dashboard":
    st.header("Welcome to Your Personal Fitness Tracker")
    st.header("Step into Your Stronger Self!")
    st.write("Ready to elevate your fitness experience? This dashboard is designed to inspire and guide you. Explore your personalized path to wellness, celebrate your achievements, and fuel your motivation. Your journey to a healthier, happier you begins right here!")
    st.write("Here's what you can do:")
    st.write("- Predict calories burned during exercise.")
    st.write("- Get personalized AI exercise suggestions.")
    st.write("- Track your progress with interactive charts.")
    st.write("- Calculate BMI.")
    
elif page == "Calorie Prediction":
    st.header("Calorie Prediction")
    # Reset total calories burned when entering the page
    st.session_state.total_calories_burned = 0

    st.sidebar.header("User Input Parameters:")

    def user_input_features():
        age = st.sidebar.slider("Age: ", 10, 100, 30)
        bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
        duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
        heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
        gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

        gender = 1 if gender_button == "Male" else 0

        exercise_type = st.sidebar.selectbox("Exercise Type:", ["Running", "Cycling", "Swimming", "Weightlifting", "Yoga"])

        data_model = {
            "Age": age,
            "BMI": bmi,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Gender_male": gender,
            "Exercise_Type": exercise_type,
        }

        features = pd.DataFrame(data_model, index=[0])
        return features

    df_user_input = user_input_features()
    df = df_user_input.copy()

    st.write("---")
    st.header("Your Parameters: ")
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    st.write(df_user_input)

    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)
    exercise_df['Date'] = pd.to_datetime(pd.Series(pd.date_range(end=datetime.datetime.now(), periods=len(exercise_df)).date))

    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Calories"]]
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]

    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]

    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)

    df = df.reindex(columns=X_train.columns, fill_value=0)
    prediction = random_reg.predict(df)

    st.write("---")
    st.header("Prediction: ")
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

    st.write(f"**Predicted Calories:** {round(prediction[0], 2)} kilocalories")

    st.write("---")
    st.header("Similar Results: ")
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    st.write(similar_data.sample(5))

   # Store data in session state and update calories burned
    st.session_state.exercise_df = exercise_df
    st.session_state.df_user_input = df_user_input
    st.session_state.total_calories_burned += prediction[0]

elif page == "AI Suggestions":
    st.header("AI-Powered Exercise Suggestions")

    st.sidebar.header("User Input Parameters:")

    def user_input_features_ai():
        age = st.sidebar.slider("Age: ", 10, 100, 30)
        duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
        gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
        gender = 1 if gender_button == "Male" else 0
        exercise_type = st.sidebar.selectbox("Exercise Type:", ["Running", "Cycling", "Swimming", "Weightlifting", "Yoga"])
        goal = st.sidebar.selectbox("Fitness Goal:", ["Weight Loss", "Muscle Gain", "Endurance"])
        data_model = {
            "Age": age,
            "Duration": duration,
            "Gender_male": gender,
            "Exercise_Type": exercise_type,
            "Goal": goal
        }
        features = pd.DataFrame(data_model, index=[0])
        return features

    df_user_input_ai = user_input_features_ai()

    if st.button("Get Suggestions"):
        suggestions = generate_ai_suggestions(
            df_user_input_ai["Age"][0],
            df_user_input_ai["Duration"][0],
            df_user_input_ai["Gender_male"][0],
            df_user_input_ai["Exercise_Type"][0],
            df_user_input_ai["Goal"][0]
        )
        st.write("## AI Suggestions:")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

elif page == "BMI Calculator":
    st.header("BMI Calculator")
    st.markdown(
        """
        <style>
        .stNumberInput label {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    height = st.number_input("Enter your height (cm):", min_value=10, max_value=300, value=170)
    weight = st.number_input("Enter your weight (kg):", min_value=1, max_value=300, value=70)

    if st.button("Calculate BMI"):
        height_m = height / 100  # Convert cm to meters
        bmi = weight / (height_m ** 2)
        st.write(f"Your BMI is: {bmi:.2f}")

        # Interpret BMI
        if bmi < 18.5:
            st.write("You are underweight.")
        elif 18.5 <= bmi < 25:
            st.write("You have a normal weight.")
        elif 25 <= bmi < 30:
            st.write("You are overweight.")
        else:
            st.write("You are obese.")

elif page == "Progress Tracking":
    st.header("Progress Tracking")

    if st.session_state.exercise_df is not None:
        st.subheader("Your Exercise Progress")
        
        # Calories vs. Duration
        fig_duration = px.bar(st.session_state.exercise_df, x='Duration', y='Calories', title="Calories Burned vs. Duration")
        st.plotly_chart(fig_duration)
        # Calories vs. Gender
        fig_gender = px.bar(st.session_state.exercise_df, x='Gender', y='Calories', title="Calories Burned vs. Gender")
        st.plotly_chart(fig_gender)
 

        dynamic_goal = max(st.session_state.total_calories_burned * 1.5, 200)
        progress_percentage = min(100, (st.session_state.total_calories_burned / dynamic_goal) * 100)
        if progress_percentage >= 100: st.success("Goal reached!")
        elif progress_percentage >= 75: st.info("Almost there!")
        elif progress_percentage >= 50: st.info("Good progress!")
        else: st.info("Keep going!")
        st.write(f"Calories Burned: {round(st.session_state.total_calories_burned, 2)} / {round(dynamic_goal, 2)}")
        st.plotly_chart(create_progress_gauge(progress_percentage))
    else:
        st.write("Please make a prediction first.")