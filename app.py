from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import torch.nn.functional as F
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recycle.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Update model initialization
MODEL_PATH = 'models/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the exact class mapping that matches your training data
CLASS_MAPPING = {
    0: 'plastic',    # Reordered to match training data
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'cardboard',
    5: 'trash',
    6: 'battery',
    7: 'organic',
    8: 'clothes',
    9: 'ewaste'
}

# Initialize model with correct architecture
model = models.resnet50(weights=None)
num_classes = len(CLASS_MAPPING)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, num_classes)
)

# Load model weights
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model = model.to(device)
model.eval()

# Define the classes with descriptions
WASTE_CATEGORIES = {
    'glass': {
        'recyclable': True,
        'points': 20,
        'description': 'Glass containers and bottles',
        'instructions': ['Remove caps and lids', 'Rinse thoroughly', 'Sort by color if required']
    },
    'paper': {
        'recyclable': True,
        'points': 15,
        'description': 'Clean paper products',
        'instructions': ['Keep dry', 'Remove any plastic', 'Flatten if possible']
    },
    'cardboard': {
        'recyclable': True,
        'points': 15,
        'description': 'Cardboard boxes and packaging',
        'instructions': ['Break down boxes', 'Remove tape', 'Keep dry']
    },
    'meatal': {
        'recyclable': True,
        'points': 10,
        'description': 'Plastic containers and bottles',
        'instructions': ['Rinse clean', 'Remove caps', 'Crush to save space']
    },
    'plastic': {
        'recyclable': True,
        'points': 25,
        'description': 'Metal cans and containers',
        'instructions': ['Rinse clean', 'Remove labels if possible', 'Crush if possible']
    },
    'trash': {
        'recyclable': False,
        'points': 0,
        'description': 'Non-recyclable waste',
        'instructions': ['Dispose in regular waste']
    },
    'organic': {
        'recyclable': False,
        'points': 0,
        'description': 'Food waste and organic materials',
        'instructions': ['Consider composting']
    },
    'battery': {
        'recyclable': True,
        'points': 30,
        'description': 'Used batteries',
        'instructions': ['Do not crush', 'Keep dry', 'Take to recycling center']
    },
    'clothes': {
        'recyclable': True,
        'points': 20,
        'description': 'Textile waste',
        'instructions': ['Must be clean', 'Pair shoes if applicable', 'Tie strings together']
    },
    'ewaste': {
        'recyclable': True,
        'points': 40,
        'description': 'Electronic waste',
        'instructions': ['Keep all parts together', 'Remove batteries', 'Handle with care']
    }
}

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    points = db.Column(db.Integer, default=0)
    items = db.relationship('RecycleItem', backref='user', lazy=True)

class RecycleItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, completed
    points = db.Column(db.Integer, default=0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class RecycleCenter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    contact = db.Column(db.String(50), nullable=False)
    categories = db.Column(db.String(200), nullable=False)  # Comma-separated categories

class Withdrawal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    points = db.Column(db.Integer, nullable=False)
    money_value = db.Column(db.Float, nullable=False)
    payment_method = db.Column(db.String(50), nullable=False)
    payment_details = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='pending')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='withdrawals')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def classify_image(image_path):
    try:
        # Enhanced preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Add color augmentation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Add multiple crop predictions for better accuracy
        crops = [
            transform(image).unsqueeze(0),  # Center crop
            transform(image.transpose(Image.ROTATE_90)).unsqueeze(0),  # Rotate 90
            transform(image.transpose(Image.ROTATE_180)).unsqueeze(0),  # Rotate 180
            transform(image.transpose(Image.ROTATE_270)).unsqueeze(0),  # Rotate 270
        ]
        
        predictions = []
        confidences = []
        
        # Get predictions for each crop
        with torch.no_grad():
            for crop in crops:
                crop = crop.to(device)
                outputs = model(crop)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predictions.append(predicted.item())
                confidences.append(confidence.item())
                
                # Print debug information
                print(f"\nCrop prediction: {CLASS_MAPPING[predicted.item()]}")
                print(f"Confidence: {confidence.item():.4f}")
                
                # Print top 3 predictions for this crop
                top_probs, top_indices = torch.topk(probabilities, 3)
                print("Top 3 predictions for this crop:")
                for i in range(3):
                    class_idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    print(f"{CLASS_MAPPING[class_idx]}: {prob:.4f}")
        
        # Use majority voting with confidence weighting
        prediction_counts = {}
        for pred, conf in zip(predictions, confidences):
            class_name = CLASS_MAPPING[pred]
            prediction_counts[class_name] = prediction_counts.get(class_name, 0) + conf
        
        # Get the final prediction
        final_prediction = max(prediction_counts.items(), key=lambda x: x[1])[0]
        final_confidence = prediction_counts[final_prediction] / len(crops)
        
        print(f"\nFinal prediction: {final_prediction}")
        print(f"Final confidence: {final_confidence:.4f}")
        
        # Only return prediction if confidence is high enough
        if final_confidence > 0.65:  # Adjusted threshold
            return final_prediction
        else:
            print(f"Low confidence ({final_confidence:.4f}), defaulting to trash")
            return 'trash'
                
    except Exception as e:
        print(f"Error in classification: {e}")
        print("Stack trace:", traceback.format_exc())
        return 'trash'

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form['email']  # Changed from username to email
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully!')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
            
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    items = RecycleItem.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', items=items)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded')
            return redirect(request.url)
            
        image = request.files['image']
        if image.filename == '':
            flash('No image selected')
            return redirect(request.url)
            
        if image:
            try:
                # Save the uploaded image
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                
                # Store the relative path for display
                relative_path = os.path.join('uploads', filename)
                
                # Get model prediction
                print(f"\nProcessing image: {filepath}")
                category = classify_image(filepath)
                print(f"Final category decision: {category}")
                
                category_info = WASTE_CATEGORIES[category]
                
                # Create RecycleItem with relative path
                item = RecycleItem(
                    image_path=relative_path,  # Store relative path
                    category=category,
                    points=category_info['points'],
                    user_id=current_user.id,
                    status='approved' if category_info['recyclable'] else 'rejected'
                )
                db.session.add(item)
                
                if category_info['recyclable']:
                    current_user.points += category_info['points']
                    flash(f'Successfully identified as {category}! You earned {category_info["points"]} points!')
                else:
                    if category == 'organic':
                        flash('This appears to be organic waste. Consider composting!')
                    else:
                        flash('This item appears to be non-recyclable.')
                
                db.session.commit()
                
                # Get nearby centers
                centers = RecycleCenter.query.filter(
                    RecycleCenter.categories.contains(category)
                ).all() if category_info['recyclable'] else []
                
                return render_template('result.html',
                                     category=category,
                                     points=category_info['points'],
                                     centers=centers,
                                     is_recyclable=category_info['recyclable'],
                                     item=item,
                                     image_path=relative_path,  # Pass the image path
                                     description=category_info['description'],
                                     instructions=category_info['instructions'])
                                     
            except Exception as e:
                print(f"Error in upload route: {e}")
                print("Stack trace:", traceback.format_exc())
                flash('An error occurred while processing the image. Please try again.')
                return redirect(url_for('upload'))
            
    return render_template('upload.html')

@app.route('/withdraw', methods=['GET', 'POST'])
@login_required
def withdraw():
    if request.method == 'POST':
        points_to_withdraw = int(request.form.get('points', 0))
        payment_method = request.form.get('payment_method')
        payment_details = request.form.get('payment_details')
        
        # Validate withdrawal amount
        if points_to_withdraw <= 0:
            flash('Please enter a valid number of points to withdraw.')
            return redirect(url_for('withdraw'))
            
        if points_to_withdraw > current_user.points:
            flash('You do not have enough points for this withdrawal.')
            return redirect(url_for('withdraw'))
            
        if not payment_details:
            flash('Please enter your payment details.')
            return redirect(url_for('withdraw'))
        
        # Calculate money value (example: 100 points = $1)
        money_value = points_to_withdraw / 100
        
        try:
            # Deduct points from user's account
            current_user.points -= points_to_withdraw
            
            # Create withdrawal record (you might want to create a Withdrawal model)
            withdrawal = Withdrawal(
                user_id=current_user.id,
                points=points_to_withdraw,
                money_value=money_value,
                payment_method=payment_method,
                payment_details=payment_details,
                status='pending'
            )
            db.session.add(withdrawal)
            db.session.commit()
            
            flash(f'Withdrawal of {points_to_withdraw} points (${money_value:.2f}) has been processed!')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during withdrawal. Please try again.')
            return redirect(url_for('withdraw'))
            
    return render_template('withdraw.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.')
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 