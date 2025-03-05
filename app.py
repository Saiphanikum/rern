from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_migrate import Migrate
import os
from pathlib import Path
from sqlalchemy import extract
from werkzeug.utils import secure_filename
from os import environ

app = Flask(__name__)
app.config['SECRET_KEY'] = environ.get('SECRET_KEY') or 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DATABASE_URL').replace('postgres://', 'postgresql://') if environ.get('DATABASE_URL') else 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Ensure template directory exists
if not os.path.exists('templates'):
    os.makedirs('templates')

# Ensure static directory exists
if not os.path.exists('static/css'):
    os.makedirs('static/css')

# Models
class WasteReward(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    waste_type = db.Column(db.String(50), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    points = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

class Withdrawal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    points = db.Column(db.Integer, nullable=False)
    payment_method = db.Column(db.String(50), nullable=False)
    account_details = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='pending')
    date = db.Column(db.DateTime, default=datetime.utcnow)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())
    is_admin = db.Column(db.Boolean, default=False)
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    
    # Relationships
    rewards = db.relationship('WasteReward', backref='user', lazy=True)
    withdrawals = db.relationship('Withdrawal', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

    def get_total_points(self):
        # Calculate total earned points
        earned_points = db.session.query(db.func.sum(WasteReward.points))\
            .filter_by(user_id=self.id).scalar() or 0
        
        # Calculate total withdrawn points (only completed or pending withdrawals)
        withdrawn_points = db.session.query(db.func.sum(Withdrawal.points))\
            .filter(Withdrawal.user_id == self.id,
                   Withdrawal.status.in_(['Completed', 'Pending'])).scalar() or 0
        
        return earned_points - withdrawn_points

    def get_total_waste(self):
        total = db.session.query(db.func.sum(WasteReward.weight))\
            .filter_by(user_id=self.id).scalar() or 0
        return round(total, 2)

    def get_available_points(self):
        # Total earned points
        earned_points = db.session.query(db.func.sum(WasteReward.points))\
            .filter_by(user_id=self.id).scalar() or 0
        
        # Points in pending withdrawals
        pending_points = db.session.query(db.func.sum(Withdrawal.points))\
            .filter(Withdrawal.user_id == self.id,
                   Withdrawal.status == 'Pending').scalar() or 0
        
        return earned_points - pending_points

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    print("Accessing login route")  # Debug print
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    try:
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False
            
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password_hash, password):
                login_user(user, remember=remember)
                flash('Successfully logged in!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.', 'error')
        
        print("Rendering login template")  # Debug print
        return render_template('auth/login.html')
    except Exception as e:
        print(f"Error in login route: {str(e)}")  # Debug print
        flash('An error occurred. Please try again.', 'error')
        return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    print("Accessing register route")  # Debug print
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    try:
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if not all([username, email, password, confirm_password]):
                flash('Please fill in all fields.', 'error')
                return render_template('auth/register.html')
                
            if password != confirm_password:
                flash('Passwords do not match.', 'error')
                return render_template('auth/register.html')
                
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'error')
                return render_template('auth/register.html')
                
            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'error')
                return render_template('auth/register.html')
                
            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password, method='sha256')
            )
            
            try:
                db.session.add(new_user)
                db.session.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                flash('Registration failed. Please try again.', 'error')
                return render_template('auth/register.html')
        
        print("Rendering register template")  # Debug print
        return render_template('auth/register.html')
    except Exception as e:
        print(f"Error in register route: {str(e)}")  # Debug print
        flash('An error occurred. Please try again.', 'error')
        return render_template('auth/register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if not current_user.is_authenticated:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))
    
    # Calculate total waste and points
    total_waste = current_user.get_total_waste()
    total_points = current_user.get_total_points()
    
    # Calculate monthly progress (waste uploaded this month)
    current_month = datetime.utcnow().month
    monthly_waste = db.session.query(db.func.sum(WasteReward.weight))\
        .filter(WasteReward.user_id == current_user.id,
               extract('month', WasteReward.date) == current_month).scalar() or 0
    
    user_data = {
        'total_waste': total_waste,
        'total_points': total_points,
        'monthly_progress': round(monthly_waste, 2)
    }
    
    return render_template('dashboard/dashboard.html', user=current_user, **user_data)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET'])
@login_required
def profile():
    # Calculate total waste
    total_waste = current_user.get_total_waste()
    
    # Get number of contributions (waste uploads)
    contributions = WasteReward.query.filter_by(user_id=current_user.id).count()
    
    return render_template('dashboard/profile.html', 
                         user=current_user,
                         total_waste=total_waste,
                         contributions=contributions)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    try:
        # Get form data
        phone = request.form.get('phone')
        address = request.form.get('address')
        
        # Update user
        current_user.phone = phone
        current_user.address = address
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error updating profile. Please try again.', 'error')
    
    return redirect(url_for('profile'))

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    try:
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Verify current password
        if not check_password_hash(current_user.password_hash, current_password):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('profile'))
        
        # Check if new passwords match
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return redirect(url_for('profile'))
        
        # Update password
        current_user.password_hash = generate_password_hash(new_password, method='sha256')
        db.session.commit()
        flash('Password updated successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash('Error updating password. Please try again.', 'error')
    
    return redirect(url_for('profile'))

@app.route('/rewards')
@login_required
def rewards():
    # Get user's rewards
    rewards = WasteReward.query.filter_by(user_id=current_user.id)\
        .order_by(WasteReward.date.desc()).all()
    
    return render_template('dashboard/rewards.html',
                         rewards=rewards,
                         total_points=current_user.get_total_points(),
                         available_points=current_user.get_available_points(),
                         total_waste=current_user.get_total_waste(),
                         pending_points=0)  # Update this if you track pending rewards

@app.route('/withdraw')
@login_required
def withdraw():
    # Get user's withdrawals
    withdrawals = Withdrawal.query.filter_by(user_id=current_user.id)\
        .order_by(Withdrawal.date.desc()).all()
    
    return render_template('dashboard/withdraw.html',
                         withdrawals=withdrawals,
                         available_points=current_user.get_available_points(),
                         total_points=current_user.get_total_points(),
                         total_waste=current_user.get_total_waste())

@app.route('/withdraw_points', methods=['POST'])
@login_required
def withdraw_points():
    try:
        points = int(request.form.get('points', 0))
        payment_method = request.form.get('payment_method')
        account_details = request.form.get('account_details')
        
        if not all([points, payment_method, account_details]):
            flash('Please fill in all required fields.', 'error')
            return redirect(url_for('withdraw'))
        
        available_points = current_user.get_available_points()
        
        if points > available_points:
            flash('Insufficient points available.', 'error')
            return redirect(url_for('withdraw'))
            
        if points < 100:
            flash('Minimum withdrawal amount is 100 points.', 'error')
            return redirect(url_for('withdraw'))
        
        new_withdrawal = Withdrawal(
            user_id=current_user.id,
            points=points,
            payment_method=payment_method,
            account_details=account_details,
            status='Pending'
        )
        
        db.session.add(new_withdrawal)
        db.session.commit()
        flash('Withdrawal request submitted successfully!', 'success')
        
    except ValueError:
        flash('Please enter a valid points amount.', 'error')
    except Exception as e:
        db.session.rollback()
        flash('Error processing withdrawal. Please try again.', 'error')
    
    return redirect(url_for('withdraw'))

@app.route('/add_reward', methods=['POST'])
@login_required
def add_reward():
    if not current_user.is_admin:
        flash('Unauthorized access', 'error')
        return redirect(url_for('rewards'))

    try:
        new_reward = WasteReward(
            user_id=request.form.get('user_id', type=int),
            waste_type=request.form.get('waste_type'),
            weight=request.form.get('weight', type=float),
            points=request.form.get('points', type=int),
            date=datetime.utcnow()
        )
        db.session.add(new_reward)
        db.session.commit()
        flash('Reward added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding reward: {str(e)}', 'error')
    
    return redirect(url_for('rewards'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_waste', methods=['GET', 'POST'])
@login_required
def upload_waste():
    print("Accessing upload_waste route")  # Debug print
    if request.method == 'POST':
        try:
            # Get form data
            waste_type = request.form.get('waste_type')
            quantity = float(request.form.get('quantity', 0))
            notes = request.form.get('notes', '')
            
            # Validate inputs
            if not waste_type or quantity <= 0:
                flash('Please provide valid waste type and quantity.', 'error')
                return redirect(url_for('upload_waste'))

            # Handle file upload
            if 'waste_image' not in request.files:
                flash('No file uploaded.', 'error')
                return redirect(url_for('upload_waste'))
            
            file = request.files['waste_image']
            if file.filename == '':
                flash('No file selected.', 'error')
                return redirect(url_for('upload_waste'))

            # Calculate points based on waste type
            points_per_kg = {
                'plastic': 10,
                'metal': 15,
                'glass': 12,
                'paper': 8,
                'electronic': 20
            }
            points = int(quantity * points_per_kg.get(waste_type, 0))

            # Create new waste reward
            new_reward = WasteReward(
                user_id=current_user.id,
                waste_type=waste_type,
                weight=quantity,
                points=points,
                date=datetime.utcnow()
            )
            
            # Save to database
            db.session.add(new_reward)
            db.session.commit()
            
            flash(f'Successfully uploaded waste and earned {points} points!', 'success')
            return redirect(url_for('rewards'))
            
        except ValueError as e:
            print(f"ValueError: {str(e)}")  # Debug print
            flash('Please enter valid quantity.', 'error')
            return redirect(url_for('upload_waste'))
        except Exception as e:
            print(f"Error uploading waste: {str(e)}")  # Debug print
            db.session.rollback()
            flash('Error uploading waste. Please try again.', 'error')
            return redirect(url_for('upload_waste'))
            
    # GET request - show upload form
    return render_template('dashboard/upload_waste.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

def create_directories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directories = [
        os.path.join(base_dir, 'templates'),
        os.path.join(base_dir, 'templates/auth'),
        os.path.join(base_dir, 'templates/dashboard'),
        os.path.join(base_dir, 'static/css'),
        os.path.join(base_dir, 'static/uploads')
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    create_directories()
    init_db()
    app.run(host='0.0.0.0', port=port)
