import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# Streamlit App title and description
st.title("Depression Detection System")
st.write("This application offers two ways to evaluate your depression status:")
st.write("1. Real-time emotion detection through webcam.")
st.write("2. PHQ-9 questionnaire-based assessment.")

# Load the model architecture from a JSON file and weights for emotion detection
model_json_file = 'cnn.json'  # Path to your saved model architecture
model_weights_file = 'cnn.h5'  # Path to your saved model weights

# Load the model architecture
with open(model_json_file, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)
st.sidebar.success("Model loaded successfully for real-time emotion detection")

# Define the emotion labels corresponding to the model's output classes
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the mapping of emotions to broader categories
depression_mapping = {
    'angry': 'Depressed',
    'disgust': 'Depressed',
    'fear': 'Depressed',
    'happy': 'Non-Depressed',
    'neutral': 'Non-Depressed',
    'sad': 'Depressed',
    'surprise': 'Depressed'
}

# Load Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture and process the webcam feed
def emotion_detector():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Create an empty frame in Streamlit to display the video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Extract the face region of interest (ROI)
            roi_gray = gray[y:y + h, x:x + w]

            # Resize the face region to 48x48 pixels (required input size for the model)
            roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Preprocess the face region for model input
            roi_resized = roi_resized.astype('float32') / 255.0  # Normalize pixel values
            roi_resized = np.expand_dims(roi_resized, axis=-1)  # Add channel dimension (for grayscale)
            roi_resized = np.expand_dims(roi_resized, axis=0)   # Add batch dimension

            # Make emotion prediction
            pred = model.predict(roi_resized)
            pred_label = emotion_labels[np.argmax(pred)]  # Get the predicted emotion label

            # Map the predicted label to 'Depressed' or 'Non-Depressed'
            depression_status = depression_mapping.get(pred_label, "Unknown")

            # Display the predicted depression status on the frame
            label_position = (x, y - 10)
            cv2.putText(frame, depression_status, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the BGR frame to RGB (required by Streamlit)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

# Function for the PHQ-9 Questionnaire
def phq9_questionnaire():
    st.subheader("PHQ-9 Depression Screening Questionnaire")

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

    # Interpret the score for binary classification
    def interpret_binary_classification(score):
        # Define threshold (e.g., score of 10 or above = Depressed)
        threshold = 10
        if score >= threshold:
            return "Depressed"
        else:
            return "Non-depressed"

    # Instructions
    st.write("### Please answer the following questions based on how you have felt over the last few weeks:")

    # Create a form for user inputs
    total_score = 0
    with st.form("phq9_form"):
        # Loop through the questions and display them with response options
        for i, question in enumerate(questions):
            st.write(f"**{question}**")
            # Set index to None to have no default selection
            answer = st.radio(
                f"Question {i + 1}",
                options=["Not at all", "Several days", "Over half the days", "Nearly every day"],
                index=None,  # No default selected option
                key=f"q{i}"  # Key for Streamlit to track each question's response
            )
            # Convert the answer into a score (0 to 3) if an answer is selected
            if answer:
                total_score += list(options.values()).index(answer)

        # Submit button
        submitted = st.form_submit_button("Submit")

    # If the form is submitted, show the results
    if submitted:
        st.write(f"### Your Total Score: {total_score}")
        diagnosis = interpret_binary_classification(total_score)
        st.write(f"### Diagnosis: {diagnosis}")

# Sidebar for user to select between Emotion Detection and PHQ-9
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Emotion Detection", "PHQ-9 Questionnaire"])

# Load the respective function based on user selection
if selection == "Emotion Detection":
    st.subheader("Real-Time Emotion Detection")
    if st.button('Start Webcam Emotion Detection'):
        emotion_detector()

elif selection == "PHQ-9 Questionnaire":
    phq9_questionnaire()
