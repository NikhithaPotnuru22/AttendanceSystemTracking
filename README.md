👨‍💻 Face Recognition Attendance System
A complete end-to-end face recognition-based attendance system built using Python, OpenCV, MediaPipe, scikit-learn, and Streamlit — with a focus on real software design using OOP principles.

🔧 Features
📸 Real-time face detection using MediaPipe's 468-point face mesh

🧠 Landmark-based recognition with geometry normalization

🌲 Random Forest Classifier for face prediction

✅ Login/Logout system with confirmation prompts

🧾 Attendance saved to CSV with login/logout time & work hours

🧱 Modular OOP design using Python classes

⚡ Optimized model retraining (only after registration)

🧼 Clean Streamlit UI for camera preview, feedback, and interactivity

🧠 Tech Stack
- Python 3
- OpenCV – webcam capture and image processing
- MediaPipe – facial landmark extraction
- scikit-learn – model training (RandomForest)
- Streamlit – frontend UI
- Pandas – data handling and CSV logging

📦 How It Works
- Register a face with webcam (200 samples stored)
- Model retrains and stores facial geometry embeddings
- On Login, the system:
- Captures face
- Predicts user identity
- Asks for confirmation
- Saves login time
- On Logout, it:
- Identifies user
- Saves logout time
- Calculates work hours

📁 Output
All attendance data is saved in AttendanceSystem.csv with:
- Name
- Login Time
- Login status
- Logout Time
- Work Duration
