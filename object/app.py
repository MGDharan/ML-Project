from flask import Flask, render_template
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # Use the PORT env var if provided (common in cloud hosts).
    port = int(os.environ.get('PORT', 5000))
    # Disable the reloader and debug mode in hosted environments (Streamlit / some cloud
    # containers) because Werkzeug's reloader registers signal handlers that may not be
    # permitted and can raise errors like the one you saw.
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

