import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from functools import wraps

# Initialize Flask app
app = Flask(__name__)

# Load configuration
from config import config
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch CNN-LSTM Model Definition
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        
        # LSTM layer
        self.lstm = nn.LSTM(64, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, features)
        batch_size, time_steps, features = x.size()
        
        # Reshape for CNN: (batch_size, features, time_steps)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        
        # Reshape back for LSTM: (batch_size, time_steps//2, 64)
        x = x.transpose(1, 2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        x = lstm_out[:, -1, :]
        
        # Classification head
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

# Load the trained PyTorch model and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    # Load scaler
    scaler = joblib.load(app.config['SCALER_PATH'])
    
    # Load model - use .pth extension for PyTorch
    model_path = app.config['MODEL_PATH']
    if os.path.exists(model_path):
        # Initialize model with correct input size (32 features)
        model = CNNLSTMModel(input_size=32).to(device)
        
        # Load the saved model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's a checkpoint with model_state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        logger.info("PyTorch model and scaler loaded successfully")
    else:
        logger.warning(f"PyTorch model file not found: {model_path}")
        model = None
        scaler = None
except Exception as e:
    logger.error(f"Error loading PyTorch model: {e}")
    logger.info("Using fallback prediction method")
    model = None
    scaler = None

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    readings = db.relationship('Reading', backref='patient', lazy=True, order_by='Reading.timestamp.desc()')

class Reading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Vital signs
    hr = db.Column(db.Float)
    o2sat = db.Column(db.Float)
    temp = db.Column(db.Float)
    sbp = db.Column(db.Float)
    map = db.Column(db.Float)
    dbp = db.Column(db.Float)
    resp = db.Column(db.Float)
    etco2 = db.Column(db.Float)
    hco3 = db.Column(db.Float)
    fio2 = db.Column(db.Float)
    ph = db.Column(db.Float)
    paco2 = db.Column(db.Float)
    sao2 = db.Column(db.Float)
    
    # Lab values
    bun = db.Column(db.Float)
    calcium = db.Column(db.Float)
    chloride = db.Column(db.Float)
    creatinine = db.Column(db.Float)
    bilirubin_direct = db.Column(db.Float)
    glucose = db.Column(db.Float)
    lactate = db.Column(db.Float)
    magnesium = db.Column(db.Float)
    phosphate = db.Column(db.Float)
    potassium = db.Column(db.Float)
    troponini = db.Column(db.Float)
    hct = db.Column(db.Float)
    hgb = db.Column(db.Float)
    ptt = db.Column(db.Float)
    wbc = db.Column(db.Float)
    fibrinogen = db.Column(db.Float)
    platelets = db.Column(db.Float)
    
    # Patient demographics
    age = db.Column(db.Integer)
    gender = db.Column(db.Integer)  # 0 for female, 1 for male
    
    # Prediction results
    prediction = db.Column(db.Boolean)
    confidence = db.Column(db.Float)
    prediction_source = db.Column(db.String(20))  # 'model' or 'fallback'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('register.html')
        
        user = User(
            email=email,
            name=name,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
@doctor_required
def dashboard():
    patients = Patient.query.filter_by(doctor_id=current_user.id).all()
    return render_template('dashboard.html', patients=patients)

@app.route('/patient/new', methods=['GET', 'POST'])
@doctor_required
def new_patient():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        patient = Patient(
            doctor_id=current_user.id,
            name=name,
            age=age,
            gender=gender
        )
        
        db.session.add(patient)
        db.session.commit()
        
        flash('Patient added successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('new_patient.html')

@app.route('/patient/<int:patient_id>')
@doctor_required
def patient_detail(patient_id):
    patient = Patient.query.filter_by(id=patient_id, doctor_id=current_user.id).first_or_404()
    readings = patient.readings
    return render_template('patient_detail.html', patient=patient, readings=readings)

@app.route('/patient/<int:patient_id>/reading/new', methods=['GET', 'POST'])
@doctor_required
def new_reading(patient_id):
    patient = Patient.query.filter_by(id=patient_id, doctor_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        # Get form data for all fields
        reading_data = {
            # Vital signs
            'hr': float(request.form.get('hr', 0)),
            'o2sat': float(request.form.get('o2sat', 0)),
            'temp': float(request.form.get('temp', 0)),
            'sbp': float(request.form.get('sbp', 0)),
            'map': float(request.form.get('map', 0)),
            'dbp': float(request.form.get('dbp', 0)),
            'resp': float(request.form.get('resp', 0)),
            'etco2': float(request.form.get('etco2', 0)),
            'hco3': float(request.form.get('hco3', 0)),
            'fio2': float(request.form.get('fio2', 0)),
            'ph': float(request.form.get('ph', 0)),
            'paco2': float(request.form.get('paco2', 0)),
            'sao2': float(request.form.get('sao2', 0)),
            
            # Lab values
            'bun': float(request.form.get('bun', 0)),
            'calcium': float(request.form.get('calcium', 0)),
            'chloride': float(request.form.get('chloride', 0)),
            'creatinine': float(request.form.get('creatinine', 0)),
            'bilirubin_direct': float(request.form.get('bilirubin_direct', 0)),
            'glucose': float(request.form.get('glucose', 0)),
            'lactate': float(request.form.get('lactate', 0)),
            'magnesium': float(request.form.get('magnesium', 0)),
            'phosphate': float(request.form.get('phosphate', 0)),
            'potassium': float(request.form.get('potassium', 0)),
            'troponini': float(request.form.get('troponini', 0)),
            'hct': float(request.form.get('hct', 0)),
            'hgb': float(request.form.get('hgb', 0)),
            'ptt': float(request.form.get('ptt', 0)),
            'wbc': float(request.form.get('wbc', 0)),
            'fibrinogen': float(request.form.get('fibrinogen', 0)),
            'platelets': float(request.form.get('platelets', 0)),
            
            # Patient demographics
            'age': int(request.form.get('age', patient.age)),
            'gender': int(request.form.get('gender', 0 if patient.gender == 'Female' else 1))
        }
        
        # Make prediction
        prediction, confidence, prediction_source = predict_sepsis(reading_data)
        
        # Create reading record
        reading = Reading(
            patient_id=patient_id,
            **reading_data,
            prediction=prediction,
            confidence=confidence,
            prediction_source=prediction_source
        )
        
        db.session.add(reading)
        db.session.commit()
        
        flash(f'Reading added successfully! Prediction: {"Sepsis" if prediction else "No Sepsis"} (Confidence: {confidence:.1f}%) - Source: {prediction_source.title()}', 'success')
        return redirect(url_for('patient_detail', patient_id=patient_id))
    
    return render_template('new_reading.html', patient=patient)

@app.route('/patient/<int:patient_id>/reading/<int:reading_id>/edit', methods=['GET', 'POST'])
@doctor_required
def edit_reading(patient_id, reading_id):
    patient = Patient.query.filter_by(id=patient_id, doctor_id=current_user.id).first_or_404()
    reading = Reading.query.filter_by(id=reading_id, patient_id=patient.id).first_or_404()

    if request.method == 'POST':
        # Build updated reading data from form
        updated_data = {
            'hr': float(request.form.get('hr', reading.hr or 0)),
            'o2sat': float(request.form.get('o2sat', reading.o2sat or 0)),
            'temp': float(request.form.get('temp', reading.temp or 0)),
            'sbp': float(request.form.get('sbp', reading.sbp or 0)),
            'map': float(request.form.get('map', reading.map or 0)),
            'dbp': float(request.form.get('dbp', reading.dbp or 0)),
            'resp': float(request.form.get('resp', reading.resp or 0)),
            'etco2': float(request.form.get('etco2', reading.etco2 or 0)),
            'hco3': float(request.form.get('hco3', reading.hco3 or 0)),
            'fio2': float(request.form.get('fio2', reading.fio2 or 0)),
            'ph': float(request.form.get('ph', reading.ph or 0)),
            'paco2': float(request.form.get('paco2', reading.paco2 or 0)),
            'sao2': float(request.form.get('sao2', reading.sao2 or 0)),
            'bun': float(request.form.get('bun', reading.bun or 0)),
            'calcium': float(request.form.get('calcium', reading.calcium or 0)),
            'chloride': float(request.form.get('chloride', reading.chloride or 0)),
            'creatinine': float(request.form.get('creatinine', reading.creatinine or 0)),
            'bilirubin_direct': float(request.form.get('bilirubin_direct', reading.bilirubin_direct or 0)),
            'glucose': float(request.form.get('glucose', reading.glucose or 0)),
            'lactate': float(request.form.get('lactate', reading.lactate or 0)),
            'magnesium': float(request.form.get('magnesium', reading.magnesium or 0)),
            'phosphate': float(request.form.get('phosphate', reading.phosphate or 0)),
            'potassium': float(request.form.get('potassium', reading.potassium or 0)),
            'troponini': float(request.form.get('troponini', reading.troponini or 0)),
            'hct': float(request.form.get('hct', reading.hct or 0)),
            'hgb': float(request.form.get('hgb', reading.hgb or 0)),
            'ptt': float(request.form.get('ptt', reading.ptt or 0)),
            'wbc': float(request.form.get('wbc', reading.wbc or 0)),
            'fibrinogen': float(request.form.get('fibrinogen', reading.fibrinogen or 0)),
            'platelets': float(request.form.get('platelets', reading.platelets or 0)),
            'age': int(request.form.get('age', patient.age)),
            'gender': int(request.form.get('gender', 0 if patient.gender == 'Female' else 1))
        }

        # Recompute prediction
        prediction, confidence, prediction_source = predict_sepsis(updated_data)

        # Apply updates
        for key, value in updated_data.items():
            setattr(reading, key, value)
        reading.prediction = prediction
        reading.confidence = confidence
        reading.prediction_source = prediction_source
        reading.timestamp = datetime.utcnow()

        db.session.commit()
        flash(f'Reading updated! Prediction: {"Sepsis" if prediction else "No Sepsis"} (Confidence: {confidence:.1f}%) - Source: {prediction_source.title()}', 'success')
        return redirect(url_for('patient_detail', patient_id=patient.id))

    return render_template('edit_reading.html', patient=patient, reading=reading)


@app.route('/patient/<int:patient_id>/reading/<int:reading_id>/delete', methods=['POST'])
@doctor_required
def delete_reading(patient_id, reading_id):
    patient = Patient.query.filter_by(id=patient_id, doctor_id=current_user.id).first_or_404()
    reading = Reading.query.filter_by(id=reading_id, patient_id=patient.id).first_or_404()

    db.session.delete(reading)
    db.session.commit()
    flash('Reading deleted successfully.', 'success')
    return redirect(url_for('patient_detail', patient_id=patient.id))

@app.route('/api/patient/<int:patient_id>/readings')
@doctor_required
def patient_readings_api(patient_id):
    patient = Patient.query.filter_by(id=patient_id, doctor_id=current_user.id).first_or_404()
    readings = patient.readings
    
    data = {
        'timestamps': [r.timestamp.strftime('%Y-%m-%d %H:%M') for r in readings],
        'predictions': [r.prediction for r in readings],
        'confidences': [r.confidence for r in readings],
        'temp': [getattr(r, 'temp', 0) for r in readings],
        'hr': [getattr(r, 'hr', 0) for r in readings],
        'sbp': [r.sbp for r in readings],
        'resp': [r.resp for r in readings],
        'wbc': [r.wbc for r in readings],
        'lactate': [r.lactate for r in readings],
        'o2sat': [getattr(r, 'o2sat', 0) for r in readings],
        'ph': [r.ph for r in readings]
    }
    
    return jsonify(data)

def predict_sepsis(data):
    """Make sepsis prediction using the trained model"""
    if model is None or scaler is None:
        # Fallback to simple rule-based prediction
        logger.warning("Using fallback prediction method - PyTorch model not available")
        return simple_predict_sepsis(data)
    
    try:
        logger.info("Attempting to use PyTorch model for prediction")
        # Prepare input data with all fields
        features = [
            data['hr'], data['o2sat'], data['temp'], data['sbp'], data['map'], data['dbp'],
            data['resp'], data['etco2'], data['hco3'], data['fio2'], data['ph'], data['paco2'], data['sao2'],
            data['bun'], data['calcium'], data['chloride'], data['creatinine'], data['bilirubin_direct'],
            data['glucose'], data['lactate'], data['magnesium'], data['phosphate'], data['potassium'],
            data['troponini'], data['hct'], data['hgb'], data['ptt'], data['wbc'], data['fibrinogen'],
            data['platelets'], data['age'], data['gender']
        ]
        
        # Create a 24-hour window (repeat the same values for simplicity)
        # In a real scenario, you'd use actual historical data
        window_data = np.array([features] * 24)
        
        # Scale the data
        scaled_data = scaler.transform(window_data)
        
        # Reshape for model input (batch_size, time_steps, features)
        model_input = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            prediction_prob = model(model_input).item()
        prediction = prediction_prob > 0.5
        confidence = prediction_prob * 100 if prediction else (1 - prediction_prob) * 100
        
        logger.info(f"PyTorch model prediction: {prediction}, confidence: {confidence:.2f}%")
        return bool(prediction), confidence, 'model'
        
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        logger.warning("Falling back to static prediction method")
        return simple_predict_sepsis(data)

def simple_predict_sepsis(data):
    """Simple rule-based prediction as fallback"""
    logger.info("Using static rule-based prediction method")
    score = 0
    
    # Vital signs
    if data['temp'] > 38.0 or data['temp'] < 36.0:
        score += 1
    if data['hr'] > 90:
        score += 1
    if data['sbp'] <= 100:
        score += 1
    if data['resp'] >= 22:
        score += 1
    if data['o2sat'] < 95:
        score += 1
    if data['ph'] < 7.35:
        score += 1
    
    # Lab values
    if data['wbc'] > 12.0 or data['wbc'] < 4.0:
        score += 1
    if data['lactate'] > 2.0:
        score += 1
    if data['creatinine'] > 2.0:
        score += 1
    if data['platelets'] < 150:
        score += 1
    if data['bun'] > 20:
        score += 1
    if data['hgb'] < 12:
        score += 1
    if data['bilirubin_direct'] > 1.2:
        score += 1
    if data['glucose'] > 200:
        score += 1
    if data['potassium'] > 5.5 or data['potassium'] < 3.5:
        score += 1
    if data['calcium'] < 8.5:
        score += 1
    
    prediction = score >= 4  # More conservative threshold
    confidence = min(score * 15, 95)  # Simple confidence calculation
    
    logger.info(f"Static prediction - Score: {score}, Prediction: {prediction}, Confidence: {confidence:.2f}%")
    return prediction, confidence, 'fallback'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 