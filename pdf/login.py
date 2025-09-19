import sqlite3
from flask import Flask, request, jsonify, session
from datetime import datetime, timedelta
from user_history import log_user_action, get_user_history

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database setup
def init_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )''')
    conn.commit()
    conn.close()

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    password = data.get('password')

    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (name, password) VALUES (?, ?)', (name, password))
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'User already exists'}), 400
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    name = data.get('name')
    password = data.get('password')

    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE name = ? AND password = ?', (name, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user[0]
        session['last_interaction'] = datetime.now().isoformat()  # Store as string
        return jsonify({'message': 'Login successful', 'user_id': user[0]}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    history = get_user_history(user_id)
    return jsonify({'history': history}), 200

@app.route('/log_action', methods=['POST'])
def log_action():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    action = data.get('action')
    user_id = session['user_id']
    log_user_action(user_id, action)
    return jsonify({'message': 'Action logged successfully'}), 200

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('last_interaction', None)
    return jsonify({'message': 'Logged out successfully'}), 200

@app.before_request
def session_timeout():
    if 'user_id' in session:
        if 'last_interaction' in session:
            try:
                last_interaction = datetime.fromisoformat(session['last_interaction'])
                if datetime.now() - last_interaction > timedelta(minutes=2):
                    session.pop('user_id', None)
                    session.pop('last_interaction', None)
                    return jsonify({'error': 'Session timed out'}), 401
            except Exception:
                session.pop('user_id', None)
                session.pop('last_interaction', None)
                return jsonify({'error': 'Session error'}), 401
        session['last_interaction'] = datetime.now().isoformat()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
