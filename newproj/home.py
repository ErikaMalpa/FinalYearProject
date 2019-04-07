import os
from flask import Flask, flash, request, redirect, url_for, render_template, session, abort, logging,  Response, send_file, Blueprint
from flask_socketio import SocketIO, send
from sqlalchemy import create_engine, exc
from sqlalchemy.orm import scoped_session, sessionmaker
from passlib.hash import sha256_crypt
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_wtf import FlaskForm
from wtforms.fields import StringField, SubmitField
from wtforms.validators import Required
#####
from sklearn.decomposition import PCA
import csv
import random
import math
import operator
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
##import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#####
engine = create_engine("mysql+pymysql://root:12ambionG@localhost/signup")
engine2 = create_engine("mysql+pymysql://root:12ambionG@localhost/patient")
db=scoped_session(sessionmaker(bind=engine))
db2=scoped_session(sessionmaker(bind=engine2))
c = engine2.connect()

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['txt', 'csv','soft'])

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32)
#socketio wrapper for the app
socketio = SocketIO(app,async_mode = 'eventlet')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    if 'username' in session: 
        username = session['username']
        return redirect(url_for('login'))
    else:
        return render_template("main.html")
    # if not session.get('log'):
    #     return redirect(url_for('login'))
    # else:
    #     return render_template("main.html")

@app.route('/main')
def main():
    return render_template("home.html")

@app.route('/directory')
def directory():
    return render_template("directory.html")

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
    #session["log"] = False
    session.clear()
    session.pop('username', None)
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
                    session['username'] = request.form['username']
                    session['log'] = True
                    flash(u"You are logged in","success")
                    return redirect(url_for("main"))
                else:
                    flash(u"incorrect password","danger")
                    return render_template("login.html")
    return render_template('login.html')

#######Importing necessary#######
import pylab as pl
import numpy as np
import pandas as pd
#####################################

@app.route('/convertion', methods=['GET', 'POST'])
def convertion_file():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
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
                return render_template('convertion.html')
                
        elif request.method == 'GET':
            file = 'upload/GDS6100_full.soft'
            import re

            with open(file) as f:
                data = f.read()

            match = re.search(r'\n(!dataset_table_begin\n.*?\n!dataset_table_end)\n', data, re.M | re.S)
            if match:
                with open('upload/output.csv', 'w') as f:
                    f.write(match.group(1))
            #####Read the file######
            df = pd.read_csv('upload/output.csv', sep=r'\t', skiprows=1,engine='python')

            ########List of the tiles to be removed########
            list = ['Platform_SEQUENCE','Gene title','Gene symbol','Gene ID','UniGene title','UniGene symbol','UniGene ID','Nucleotide Title','GI','GenBank Accession','Platform_CLONEID','Platform_ORF','Platform_SPOTID','Chromosome location','Chromosome annotation','GO:Function','GO:Process','GO:Component','GO:Function ID','GO:Process ID','GO:Component ID',]
            new_df = df[list]
            df = df.drop(columns=list)
            df2 = df.fillna(0)
            #df2 = df2.transpose()
            df2 = df2.to_string()
            with open('upload/converted.txt', 'w') as f:
                f.write(df2)

            return render_template('convertion.html')


# @app.route('/return-file', methods=['GET'])
# def return_file():
#       return send_file('./upload', as_attachment=True, attachment_filename="results.txt")

@app.route('/return-file')
def return_file():
    return send_file('C:\\Users\\erika\\OneDrive\\Desktop\\FinalYearProject\\newproj\\upload\\results.txt',attachment_filename='results.txt')

# @app.route('/download-file')
# def download_file():
#     if request.method == 'POST':
#         try:
#             data = request.get_json()
#             Cancer = float(data["Cancer"])

#             lin_reg = joblib.load("./linear_regression_model.pkl")
#         except ValueError:
#             return jsonify("Please enter a number.")

#         return jsonify(lin_reg.predict(years_of_experience).tolist())
#     #return render_template('download.html')

@app.route('/predict/')
def predict():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        return render_template('predict.html')

@app.route('/results/', methods=['GET', 'POST'])
def results():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        return render_template('results.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        if request.method == 'POST':
            patient_number = request.form.get("patient_number")
            age = request.form.get("age")
            gender = request.form.get("gender")
            clinic_address = request.form.get("clinic_address")
            db2.execute("INSERT INTO patient (patient_number,age,gender,clinic_address) VALUES (:patient_number, :age, :gender, :clinic_address)",
            {"patient_number":patient_number, "age":age,"gender":gender, "clinic_address":clinic_address})
            db2.commit()

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
                #with open('upload/file1.txt', 'w') as f: 
                 #   f.write(predict)
                #session["predict"] = True

                ###
#read data
                df = pd.read_csv('./upload/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                     engine='python')

                #change the 5 tumour types to numbers
                Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 

                #this is where we add the class to the table
                df.Class = [Class[item] for item in df.Class]

                #drop the 2 unnamed table because we do not need them
                df = df.drop('Unnamed: 0',1)
                df = df.drop('Unnamed: 0.1',1)

                #Split the X and y
                X = df.drop('Class', axis=1).values
                y = df['Class'].values
                y = np.asarray(y)

                #Standarize using min and max
                X = (X - X.mean()) / (X.max() - X.min())

                #Split the data set with 80 to traina dn20 to test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

                #using k=n^(1/2) where n = columns, therefore it's 143.2 
                K = 143

                clf = KNeighborsClassifier(n_neighbors=K, weights='distance')

                clf.fit(X_train, y_train)

                # df2 = pd.read_csv('./upload/test.csv', sep=r'\s*(?:\||\#|\,)\s*',
                #      engine='python')
                # df2 = np.asarray(df2)

                # df2 = clf.predict(df2)
                df2 = pd.read_csv('./upload/test.csv', sep=r'\s*(?:\||\#|\,)\s*',engine='python')
                df2 = df2.drop('Unnamed: 0', axis=1).values
                df2 = np.asarray(df2)
                df2 = clf.predict(df2)
                ####

                ####
                # patientID = request.form.get("patientID")
                # clinicAddress = request.form.get("clinicAddress")

                # patiedIDdata = db.execute("SELECT patientID FROM patientInformation WHERE username=:username",{"username":username}).fetchone()
                # clinicAddressdata = db.execute("SELECT clinicAddress FROM patientInformation WHERE username=:username",{"username":username}).fetchone()
                ####

                ####

                ####
                return render_template('results.html',predict= predict,df2=df2)
                
        elif request.method == 'GET':
            return render_template('predict.html')

@app.route('/predictOVR/')
def predictOVR():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        return render_template('predictOVR.html')

@app.route('/predictOVR', methods=['GET', 'POST'])
def upload_file2():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
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
                #with open('upload/file1.txt', 'w') as f: 
                 #   f.write(predict)
                #session["predict"] = True

                ###
                #read data
                df = pd.read_csv('./upload/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                     engine='python')

                #change the 5 tumour types to numbers
                Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 

                #this is where we add the class to the table
                df.Class = [Class[item] for item in df.Class]

                #drop the 2 unnamed table because we do not need them
                df = df.drop('Unnamed: 0',1)
                df = df.drop('Unnamed: 0.1',1)

                #Split the X and y
                X = df.drop('Class', axis=1).values
                y = df['Class'].values
                y = np.asarray(y)

                #Standarize using min and max
                X = (X - X.mean()) / (X.max() - X.min())

                #Split the data set with 80 to traina dn20 to test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

                #using k=n^(1/2) where n = columns, therefore it's 143.2 
                
                clf5 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
                lrTest = clf5.predict(X_test)

                # df2 = pd.read_csv('./upload/test.csv', sep=r'\s*(?:\||\#|\,)\s*',
                #      engine='python')
                # df2 = np.asarray(df2)

                # df2 = clf.predict(df2)
                df2 = pd.read_csv('./upload/test.csv', sep=r'\s*(?:\||\#|\,)\s*',engine='python')
                df2 = df2.drop('Unnamed: 0', axis=1).values
                df2 = np.asarray(df2)
                df2 = clf5.predict(df2)
                ####

                ####
                # patientID = request.form.get("patientID")
                # clinicAddress = request.form.get("clinicAddress")

                # patiedIDdata = db.execute("SELECT patientID FROM patientInformation WHERE username=:username",{"username":username}).fetchone()
                # clinicAddressdata = db.execute("SELECT clinicAddress FROM patientInformation WHERE username=:username",{"username":username}).fetchone()
                ####

                ####

                ####
               
                return render_template('results.html',predict= predict,df2=df2)
                
        elif request.method == 'GET':
            return render_template('predictOVR.html')

@app.route('/list')
def list():
    # c.execute("SELECT * FROM patient")
    result = engine2.execute("select * from patient")
    return render_template('list.html', result = result)
#def download():
 #   file = open('khan_train.csv','r')
  #  returnfile = file.read().encode('latin-1')
   # file.close()
    #return Response(returnfile,
     #   mimetype="text/csv",
      #  headers={"Content-disposition":
       #          "attachment; filename=khan_train.csv"})

# @app.route('/')
# def chat():
#     if not session.get('log'):
#         return redirect(url_for('login'))
#     else:
#         return render_template('chat.html')

# def messageRecived():
#   print( 'message was received!!!' )

# #event for broadcasting message to everyone...
# @socketio.on('my event')
# def handle_my_custom_event( json ):
#   print( 'recived my event: ' + str( json ) )
#   socketio.emit( 'my response', json, callback=messageRecived,room=sid )

#######
@socketio.on('joined', namespace='/chat')
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    room = session.get('room')
    join_room(room)
    emit('status', {'msg': session.get('name')}, room=room)

@socketio.on('left', namespace='/chat')
def left(message):
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session.get('room')
    leave_room(room)
    emit('status', {'msg': session.get('name')}, room=room)


@socketio.on('text', namespace='/chat')
def text(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    room = session.get('room')
    emit('message', {'msg': session.get('name') + ':' + message['msg']}, room=room)

class LoginForm(FlaskForm):
    """Accepts a nickname and a room."""
    name = StringField('Name', validators=[Required()])
    room = StringField('Room', validators=[Required()])
    submit = SubmitField('Enter Chatroom')
#########

@app.route('/index', methods=['GET', 'POST'])
def index():
    FlaskForm = LoginForm()
    if FlaskForm.validate_on_submit():
        session['name'] = FlaskForm.name.data
        session['room'] = FlaskForm.room.data
        return redirect(url_for('.chat'))
    elif request.method == 'GET':
        FlaskForm.name.data = session.get('name', '')
        FlaskForm.room.data = session.get('room', '')
    return render_template('index.html', FlaskForm=FlaskForm)
def messageRecived():
  print( 'message was received!!!' )

@app.route('/chat')
def chat():
    name = session.get('name', '')
    room = session.get('room', '')
    if name == '' or room == '':
        return redirect(url_for('.index'))
    return render_template('chat.html', name=name, room=room)

if __name__ == '__main__':
    #app.secret_key="this0is1a2pass3word4"
    #app.run(debug=True)
    socketio.run(app,debug = False, host='0.0.0.0',port=5000)
    #socketio.run(host='0.0.0.0', debug = True, app)
