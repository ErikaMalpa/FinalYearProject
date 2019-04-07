# import pytest
# import os
# from flask_socketio import SocketIO
# import app
# from flask import Flask

# @pytest.mark.options(debug=False)
# def test_app(app):
#   assert not app.debug, 'Ensure the app not in debug mode'
#
# UPLOAD_FOLDER = 'upload'

# @pytest.fixture
# def app():
#     app = Flask(__name__)
#     app.config['SECRET_KEY'] = os.urandom(32)
#     #socketio wrapper for the app
#     socketio = SocketIO(app,async_mode = 'eventlet')
#     app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#     return app


# def test_example(client):
#     response = app.get("/")
#     assert response.status_code == 200

# def test_example2(client):
#     response = client.get("/main")
#     assert response.status_code == 404

# def test_example3(client):
#     response = client.get("/directory")
#     assert response.status_code == 404

# def test_example4(client):
#     response = client.get("/logout")
#     assert response.status_code == 404


# def test_example5(client):
#     response = client.get("/login")
#     assert response.status_code == 404

from flask import Flask
from flask_testing import TestCase

class MyTest(TestCase):

    def create_app(self):

        app = Flask(__name__)
        app.config['TESTING'] = True
        return app