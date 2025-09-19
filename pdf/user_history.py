import sqlite3

def log_user_action(user_id, action):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO history (user_id, action) VALUES (?, ?)', (user_id, action))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT action, timestamp FROM history WHERE user_id = ?', (user_id,))
    history = cursor.fetchall()
    conn.close()
    return history
