import streamlit as st

# PHQ-9 Questions
questions = [
    "1. Over the last few weeks, how often have you had little interest or pleasure in doing things?",
    "2. Over the last few weeks, how often have you felt down, depressed, or hopeless?",
    "3. Over the last few weeks, how often have you had trouble falling or staying asleep, or sleeping too much?",
    "4. Over the last few weeks, how often have you felt tired or had little energy?",
    "5. Over the last few weeks, how often have you had a poor appetite or overeaten?",
    "6. Over the last few weeks, how often have you felt bad about yourself, that you’re a failure or that you’ve let yourself or your family down?",
    "7. Over the last few weeks, how often have you had trouble concentrating on things, such as reading the newspaper or watching television?",
    "8. Over the last few weeks, how often have you been moving or speaking so slowly that other people could have noticed? Or the opposite: being so fidgety or restless that you have been moving around a lot more than usual?",
    "9. Over the last few weeks, how often have you had thoughts of death or of hurting yourself?"
]

# Options for responses
options = {
    "0": "Not at all",
    "1": "Several days",
    "2": "Over half the days",
    "3": "Nearly every day"
}

# Interpret the total score
def interpret_score(score):
    if score <= 4:
        return "No depression"
    elif score <= 9:
        return "Mild depression"
    elif score <= 14:
        return "Moderate depression"
    elif score <= 19:
        return "Moderately severe depression"
    else:
        return "Severe depression"

# Streamlit app layout
st.title("PHQ-9 Depression Screening Questionnaire")

# Instructions
st.write("""
### Please answer the following questions based on how you have felt over the last few weeks:
""")

# Create a form for user inputs
total_score = 0
with st.form("phq9_form"):
    # Loop through the questions and display them with response options
    for i, question in enumerate(questions):
        st.write(f"**{question}**")
        answer = st.radio(
            f"Question {i + 1}",
            options=["Not at all", "Several days", "Over half the days", "Nearly every day"],
            index=0,  # Default selected option
            key=f"q{i}"  # Key for Streamlit to track each question's response
        )
        # Convert the answer into a score (0 to 3)
        total_score += list(options.values()).index(answer)

    # Submit button
    submitted = st.form_submit_button("Submit")

# If the form is submitted, show the results
if submitted:
    st.write(f"### Your Total Score: {total_score}")
    diagnosis = interpret_score(total_score)
    st.write(f"### Diagnosis: {diagnosis}")