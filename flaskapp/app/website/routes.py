from website import app
from flask import render_template, request, flash
from forms import ContactForm
from flask.ext.mail import Message, Mail
 
mail = Mail()

@app.route('/testdb')
def testdb():
  if db.session.query("1").from_statement("SELECT 1").all():
    return 'It works.'
  else:
    return 'Something is broken.'