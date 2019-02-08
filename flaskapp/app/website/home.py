import os
from flask import Flask, flash, request, redirect, url_for, render_template, session, abort
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def login():
	if not session.get('logged_in'):
		return render_template('login.html')
	#else:
    	#return render_template('home.html')

@app.route('/home/')
def home():
	return render_template('home.html')

@app.route("/logout/")
def logout():
	session['logged_in'] = False
	return login()

@app.route('/login', methods=['POST'])
def do_admin_login():
	if request.form['password'] == '123' and request.form['username'] == 'admin':
		session['logged_in'] = True
	else:
	   flash('wrong password!')
	return render_template('home.html')

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
	app.secret_key = os.urandom(12)
app.run(debug=True)