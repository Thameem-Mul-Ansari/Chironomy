# Sign Language Detection App

This project is a web application that utilizes computer vision to detect sign language gestures in real-time using Streamlit and TensorFlow. It captures webcam video feed and predicts sign language gestures based on pre-trained models.

## Features

- Real-time hand and pose detection using Mediapipe.
- Gesture recognition through a trained TensorFlow model.
- User-friendly interface with webcam control.

## Technologies Used

- **Streamlit**: Framework for creating web applications in Python.
- **TensorFlow**: For running the trained machine learning model.
- **Mediapipe**: For hand and pose detection.
- **OpenCV**: For video processing and drawing detected keypoints.

## Live Demo

You can access the live demo of the application at the following link:

[Hosted Sign Language Detection App](YOUR_HEROKU_OR_STREAMLIT_LINK)

## Getting Started Locally

### Prerequisites

Make sure you have the following installed on your machine:

- Python (v3.7 or later)
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/sign-language-app.git
   cd sign-language-app

2. **Install the dependencies:**
   ```bash
  pip install -r requirements.txt

3. **Run the application:**
  ```bash
  streamlit run app.py

4.Open your web browser and navigate to the provided local URL (usually http://localhost:8501) to access the application.
