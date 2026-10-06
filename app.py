from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# -----------------------------
# AI / ML TRAINING DATA
# -----------------------------

profiles = {
    "Data Scientist": [
        [5, 4, 5, 4, 3, 1],
        [4, 5, 5, 3, 3, 1],
        [5, 4, 4, 5, 3, 1]
    ],

    "Data Analyst": [
        [4, 5, 5, 3, 4, 2],
        [3, 5, 4, 3, 5, 2],
        [4, 4, 5, 3, 5, 2]
    ],

    "ML Engineer": [
        [5, 3, 4, 5, 3, 3],
        [5, 2, 4, 5, 3, 3],
        [4, 3, 4, 5, 3, 3]
    ],

    "Web Developer": [
        [5, 2, 2, 5, 4, 4],
        [4, 2, 2, 5, 4, 4],
        [4, 3, 2, 5, 5, 4]
    ],

    "Cyber Security Analyst": [
        [3, 3, 3, 4, 3, 5],
        [4, 3, 3, 4, 3, 5],
        [3, 4, 2, 4, 3, 5]
    ]
}

# -----------------------------
# CREATE TRAINING DATA
# -----------------------------

X = []
y = []

rng = np.random.default_rng(42)

for career, data in profiles.items():

    for values in data:

        for _ in range(30):

            new_values = np.clip(
                np.array(values) + rng.normal(0, 0.6, 6),
                1,
                5
            )

            X.append(new_values)
            y.append(career)

# -----------------------------
# TRAIN AI MODEL
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICTION
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get student input
    python = int(request.form["python"])
    sql = int(request.form["sql"])
    statistics = int(request.form["statistics"])
    programming = int(request.form["programming"])
    communication = int(request.form["communication"])
    interest = int(request.form["interest"])

    # Student data
    student_data = [[
        python,
        sql,
        statistics,
        programming,
        communication,
        interest
    ]]

    # Predict career
    prediction = model.predict(student_data)[0]

    # -----------------------------
    # CALCULATE SKILL SCORE
    # -----------------------------

    skill_scores = [
        python,
        sql,
        statistics,
        programming,
        communication
    ]

    average_skill = sum(skill_scores) / len(skill_scores)

    # Convert 1-5 skill level into percentage
    confidence = round((average_skill / 5) * 100, 2)

    # -----------------------------
    # FIND SKILLS TO IMPROVE
    # -----------------------------

    skills = []

    if python < 4:
        skills.append("Python")

    if sql < 4:
        skills.append("SQL")

    if statistics < 4:
        skills.append("Statistics")

    if programming < 4:
        skills.append("Programming")

    if communication < 4:
        skills.append("Communication")

    # -----------------------------
    # CAREER ROADMAP
    # -----------------------------

    roadmaps = {

        "Data Scientist":
            "Python → SQL → Statistics → Machine Learning → Deep Learning → Projects",

        "Data Analyst":
            "Excel → SQL → Python → Statistics → Power BI → Data Analytics Projects",

        "ML Engineer":
            "Python → Mathematics → Machine Learning → Deep Learning → MLOps → Projects",

        "Web Developer":
            "HTML → CSS → JavaScript → React → Backend → Full Stack Projects",

        "Cyber Security Analyst":
            "Networking → Linux → Cyber Security Basics → Ethical Hacking → Security Tools"
    }

    roadmap = roadmaps[prediction]

    # -----------------------------
    # SEND RESULT TO RESULT PAGE
    # -----------------------------

    return render_template(
        "result.html",
        career=prediction,
        confidence=confidence,
        skills=skills,
        roadmap=roadmap
    )


# -----------------------------
# RUN FLASK APPLICATION
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)