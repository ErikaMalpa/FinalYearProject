from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()


def create_app(debug=False):
    """Create an application."""
    app = Flask(__name__)
	app.config['SECRET_KEY'] = os.urandom(32)

    socketio.init_app(app)
    return app

