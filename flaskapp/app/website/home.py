import os
from flask import Flask, flash, request, redirect, url_for, render_template, session, abort, logging
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from passlib.hash import sha256_crypt
from werkzeug.utils import secure_filename

engine = create_engine("mysql+pymysql://root:12@mbionG@localhost/register")
db=scoped_session(sessionmaker(bind=engine))

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
	return render_template("main.html")

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        return render_template('login.html')
            
    elif request.method == 'GET':
        return render_template('main.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm = request.form.get("confirm")
        secure_password = sha256_crypt.encrypt(str(password))

        if password == confirm:
            db.execute("INSERT INTO users(email,username,password) VALUES (:email, :username, :password)",
                {"email":email, "username":username,"password":secure_password})
            db.commit()
            return redirect(url_for('login'))
        else:
            flash("password must match")
            return render_template("signup.html")

    return render_template('signup.html')

#@app.route("/logout/")
#def logout():
#	session['logged_in'] = False
#	return login()

@app.route('/login')
def login():
	return render_template('login.html')

@app.route('/predict/')
def predict():
    return render_template('predict.html')

@app.route('/results/')
def results():
    return render_template('results.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('results.html')
            
    elif request.method == 'GET':
    	return render_template('predict.html')
if __name__ == '__main__':
    app.run(debug=True)