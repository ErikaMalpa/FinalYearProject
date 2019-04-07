import pytest
import app
from flask import Flask

# @pytest.mark.options(debug=False)
# def test_app(app):
#   assert not app.debug, 'Ensure the app not in debug mode'

@pytest.fixture
def app():
    app = Flask(__name__)
    return app

def test_example4(client):
    response = client.get("/logout")
    assert response.status_code == 400


def test_example5(client):
    response = client.get("/login")
    assert response.status_code == 200
