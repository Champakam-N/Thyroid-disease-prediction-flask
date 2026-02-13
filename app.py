from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import json
import io
import base64

# ==========================================================
# ⚙️ Flask Setup
# ==========================================================
app = Flask(__name__)
app.secret_key = 'thyroid_flask_secret_key'


# ==========================================================
# 📦 Database Connection
# ==========================================================
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn


# ==========================================================
# 🧩 Initialize Database (if not exists)
# ==========================================================
def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );''')
    conn.commit()
    conn.close()


init_db()


# ==========================================================
# 🤖 Load CNN Model & Class Labels
# ==========================================================
model_path = r"thyroid_cnn_model_3class.h5"
model = load_model(model_path)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Ensure correct order of labels
class_labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    class_labels[idx] = label
print("✅ Loaded class labels:", class_labels)


# ==========================================================
# 🌐 ROUTES
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# ==========================================================
# 👤 Registration
# ==========================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        if not name or not email or not password:
            flash("⚠️ All fields are required.", "danger")
            return redirect(url_for('register'))

        conn = get_db_connection()
        existing_user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if existing_user:
            conn.close()
            flash("⚠️ Email already registered. Please log in.", "warning")
            return redirect(url_for('login'))

        conn.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
        conn.commit()
        conn.close()

        session.clear()  # Ensure clean session after register
        flash("✅ Registration successful! Please log in to continue.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


# ==========================================================
# 🔐 Login
# ==========================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        if not email or not password:
            flash("⚠️ Please fill in both email and password.", "danger")
            return redirect(url_for('login'))

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        # If user not found → ask to register
        if not user:
            flash("⚠️ User not found. Please register first.", "warning")
            return redirect(url_for('register'))

        # If password incorrect → stay on login page
        if user['password'] != password:
            flash("❌ Invalid password! Please try again.", "danger")
            return redirect(url_for('login'))

        # ✅ Successful login
        session.clear()  # clear any old session data
        session['logged_in'] = True
        session['email'] = email
        session['username'] = user['name']

        flash(f"👋 Welcome back, {user['name']}!", "success")
        return redirect(url_for('predict'))  # redirect to prediction

    return render_template('login.html')


# ==========================================================
# 🧠 Prediction Page
# ==========================================================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # 🔒 Step 1: Must be logged in
    if not session.get('logged_in'):
        flash("⚠️ Please log in to access the prediction feature.", "warning")
        return redirect(url_for('login'))

    # 🧠 Step 2: Handle POST request (prediction)
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("Please upload an image file.", "danger")
            return redirect(request.url)

        try:
            # Image preprocessing
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Model prediction
            preds = model.predict(img_array)[0]
            index = np.argmax(preds)
            label = class_labels[index]
            confidence = preds[index] * 100

            # Convert uploaded image to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return render_template(
                'predict.html',
                image_data=img_str,
                label=label,
                confidence=round(confidence, 2)
            )

        except Exception as e:
            flash(f"❌ Error processing image: {e}", "danger")
            return redirect(url_for('predict'))

    # GET request → just show the page
    return render_template('predict.html', image_data=None, label=None, confidence=None)


# ==========================================================
# 🚪 Logout
# ==========================================================
@app.route('/logout')
def logout():
    session.clear()
    flash("👋 You have been logged out successfully.", "info")
    return redirect(url_for('login'))


# ==========================================================
# 🛠️ Redirect Old Links (for /prediction)
# ==========================================================
@app.route('/prediction')
def prediction_redirect():
    return redirect(url_for('predict'))


# ==========================================================
# 🚀 Run Flask App
# ==========================================================
if __name__ == '__main__':
    print("✅ Flask App Running at: http://127.0.0.1:5000/")
    app.run(debug=True)
