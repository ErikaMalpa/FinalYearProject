import os
from flask import Flask, flash, request, redirect, url_for, render_template, session, abort, logging,  Response
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from passlib.hash import sha256_crypt
from werkzeug.utils import secure_filename

engine = create_engine("mysql+pymysql://root:12ambionG@localhost/signup")
db=scoped_session(sessionmaker(bind=engine))

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
	return render_template("main.html")

@app.route('/main')
def main():
    return render_template("home.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")
        #confirm = request.form.get("confirm")
        secure_password = sha256_crypt.encrypt(str(password))

        #if password == confirm:
        db.execute("INSERT INTO users (email,username,password) VALUES (:email, :username, :password)",
            {"email":email, "username":username,"password":secure_password})
        db.commit()
        return redirect(url_for('login'))
        #else:
        #    flash(u'Password must match', 'error')
        #    return render_template("signup.html")
    return render_template('signup.html')

@app.route("/logout")
def logout():
    session["log"] = False
    session.clear()
    flash("You are now logged out")
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        usernamedata = db.execute("SELECT username FROM users WHERE username=:username",{"username":username}).fetchone()
        passwordata = db.execute("SELECT password FROM users WHERE username=:username",{"username":username}).fetchone()

        if usernamedata is None:
            flash(u"No username","danger")
            return render_template("login.html")
        else:
            for password_data in passwordata:
                if sha256_crypt.verify(password,password_data):
                    session["log"] = True
                    flash(u"You are logged in","success")
                    return redirect(url_for("main"))
                else:
                    flash(u"incorrect password","danger")
                    return render_template("login.html")
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

@app.route('/predict', methods=['GET', 'POST'])
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
            predict = request.form
            #session["predict"] = True
            return render_template('results.html',predict= predict)
            
    elif request.method == 'GET':
    	return render_template('predict.html')
if __name__ == '__main__':
    app.secret_key="this0is1a2pass3word4"
    app.run(debug=True)