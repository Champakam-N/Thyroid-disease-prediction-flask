# Thyroid-disease-prediction-flask
Thyroid disease classification using CNN and Flask with user authentication
# Thyroid Disease Prediction Web Application

A web-based Thyroid Disease Classification system built using Flask and a Convolutional Neural Network (CNN).  
The application allows users to register, log in, and upload thyroid images to get disease predictions with confidence scores.

---

##  Features

- User Registration and Login (SQLite database)
- Secure session-based authentication
- Image upload and preprocessing
- CNN-based thyroid disease classification (3 classes)
- Confidence score display
- Clean Flask web interface

---

##  Machine Learning Model

- CNN architecture built using TensorFlow/Keras
- Input image size: 128x128
- Data augmentation applied
- Early stopping and model checkpoint used
- Best model saved as `.h5`

---

##  Technologies Used

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- SQLite
- HTML & CSS

---

##  Project Structure

├── app.py
├── requirements.txt
├── README.md
├── model/
│ ├── thyroid_cnn_model_3class.h5
│ └── class_indices.json
├── training/
│ └── train_model.py
├── templates/
└── static


---

##  How to Run

1. Install Python 3.10 or above  
2. Install required packages:

   pip install -r requirements.txt

3. Run the Flask application:

   python app.py

4. Open in browser:

   http://127.0.0.1:5000/

---

##  Model Training

The training script is available in the `training/` folder.  
The dataset is not included due to size limitations.

---

##  Notes

- CSS styling is written directly inside HTML templates.
- Static folder contains image assets.
- This project is developed for academic and demonstration purposes.

---

##  Author

Champa
