#### START IMPORTS ####
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
#### END IMPORTS ####

##### START IMPORTS FOR THE MODELS #######
import csv
import random
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
import pylab as pl
import tensorflow as tf
import pylab as pl
##### END IMPORTS FOR THE MODEL #####

##### MYSQL DATABASE #####

##mysql database for user sign up and log in 
engine = create_engine("mysql+pymysql://root:12ambionG@localhost/signup")

##mysql database for patients
engine2 = create_engine("mysql+pymysql://root:12ambionG@localhost/patient")

##make a session
db=scoped_session(sessionmaker(bind=engine))
db2=scoped_session(sessionmaker(bind=engine2))

##connect to database
c = engine2.connect()
##### END MYSQL DATABASE #####

##the name of the upload folder
UPLOAD_FOLDER = 'upload'

##extensions that are allowed when uploading
ALLOWED_EXTENSIONS = set(['txt', 'csv','soft'])

app = Flask(__name__)

##the secret key to make the client side secure and it is put to random
app.config['SECRET_KEY'] = os.urandom(32)

#socketio wrapper for the app
#socketio = SocketIO(app,async_mode = 'eventlet')
socketio = SocketIO(app)

##configure the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

##the index route of the application
@app.route('/')
def home():
    if 'username' in session: 
        username = session['username']
        return redirect(url_for('login'))
    else:
        return render_template("main.html")

##the route of the main page which will load the home page of the application
@app.route('/main')
def main():
    return render_template("home.html")

##The sign-up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    ##if there is a submit post
    if request.method == 'POST':
        ##get email, username and password
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")
        ##make the password secure using sha256
        secure_password = sha256_crypt.encrypt(str(password))

        ##Put it into the database
        db.execute("INSERT INTO users (email,username,password) VALUES (:email, :username, :password)",
            {"email":email, "username":username,"password":secure_password})
        ##commit to the database
        db.commit()
        ##when submitted succesfully load the login page
        return redirect(url_for('login'))
    ##else load/refresh signup page
    return render_template('signup.html')

##The log in route 
@app.route('/login', methods=['GET', 'POST'])
def login():
    ##If submitting
    if request.method == 'POST':
        ##getting the username and password
        username = request.form.get("username")
        password = request.form.get("password")

        ##Checking the inputted username and password
        usernamedata = db.execute("SELECT username FROM users WHERE username=:username",{"username":username}).fetchone()
        passwordata = db.execute("SELECT password FROM users WHERE username=:username",{"username":username}).fetchone()

        ##If there is no data then it will refresh the page
        if usernamedata is None:
            return render_template("login.html")
        else:
            ##else verify the password and decrypt the password
            for password_data in passwordata:
                if sha256_crypt.verify(password,password_data):
                    session['username'] = request.form['username']
                    ##the session log is true which will give access to users
                    session['log'] = True
                    ##It will then load the main page
                    return redirect(url_for("main"))
                else:
                    ##else it will go refresh the page
                    return render_template("login.html")
    return render_template('login.html')

##The log out route
@app.route("/logout")
def logout():
    #session["log"] = False
    ##clears the session
    session.clear()
    ##removes the username
    session.pop('username', None)
    ##goes back to home page
    return redirect(url_for('home'))

##This loads the directory route
@app.route('/directory')
def directory():
    return render_template("directory.html")

########### Start Convertion #################

##route for convertion
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
            # if user does not select a file, browser also submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                ##save uploaded fiel into folder
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                ##load convertion
                return render_template('convertion.html')
                
        elif request.method == 'GET':

            ##the sample fiel to be uploadedd
            file = 'upload/GDS6100_full.soft'

            import re

            ##we open and read the file
            with open(file) as f:
                data = f.read()

            ##if !dataset_table_begin and !dataset_table_end matches then the files inbetween will be kept
            match = re.search(r'\n(!dataset_table_begin\n.*?\n!dataset_table_end)\n', data, re.M | re.S)
            if match:
                ##it will then upload it to output.csv
                with open('upload/output.csv', 'w') as f:
                    f.write(match.group(1))

            #####Read the file######
            df = pd.read_csv('upload/output.csv', sep=r'\t', skiprows=1,engine='python')

            ########List of the tiles to be removed########
            list = ['Platform_SEQUENCE','Gene title','Gene symbol','Gene ID','UniGene title','UniGene symbol','UniGene ID','Nucleotide Title','GI','GenBank Accession','Platform_CLONEID','Platform_ORF','Platform_SPOTID','Chromosome location','Chromosome annotation','GO:Function','GO:Process','GO:Component','GO:Function ID','GO:Process ID','GO:Component ID',]
            new_df = df[list]
            df = df.drop(columns=list)
            df = df[df.IDENTIFIER != '--Control']

            #transpose
            df = df.transpose()
            df['Class'] = '1'
            df2 = df.fillna(0)

            #This is different from the main software as this is needed for it to work for flask
            #The one in the original work transforms it into a .csv file
            df2 = df2.to_string()
            ##puts it into converted.txt file
            with open('upload/converted.txt', 'w') as f:
                f.write(df2)

            ##Refreshes the convertion file
            return render_template('convertion.html')

########### END Convertion #################

##route for return file of results.txt to downlaod in the list
@app.route('/return-file')
def return_file():
    return send_file('C:\\Users\\erika\\OneDrive\\Desktop\\FinalYearProject\\newproj\\upload\\results.txt',attachment_filename='results.txt')

################ START route of the predict route of KNN ################
@app.route('/predict/')
def predict():
    ##this makes sure that the logged in users only have access to it
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        return render_template('predict.html')

from flask import Flask, request, jsonify
import pickle

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        if request.method == 'POST':

            ##The inputted file is being taken
            patient_number = request.form.get("patient_number")
            age = request.form.get("age")
            gender = request.form.get("gender")
            clinic_address = request.form.get("clinic_address")

            ##execute and insert into database
            db2.execute("INSERT INTO patient (patient_number,age,gender,clinic_address) VALUES (:patient_number, :age, :gender, :clinic_address)",
            {"patient_number":patient_number, "age":age,"gender":gender, "clinic_address":clinic_address})

            ##Commit inputted patient files into database
            db2.commit()

            ##check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also, submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                predict = request.form

            # Load the model
            model = pickle.load(open('KNN.pkl','rb'))

            df2 = pd.read_csv('./upload/test.csv', sep=r'\s*(?:\||\#|\,)\s*',engine='python')
            df2 = df2.drop('Unnamed: 0', axis=1).values
            df2 = np.asarray(df2)
            #df2 = model.predict(df2)

            with open('KNN.pkl','rb') as file:
                mp = pickle.load(file)

            df2 = mp.predict(df2)

            ##prediction = model.predict([[np.array(data['df2'])]])

            return render_template('results.html',predict= predict,df2=df2)
                
        elif request.method == 'GET':
            return render_template('predict.html')

################ END route of the predict route of KNN ################

##route of the results route
@app.route('/results/', methods=['GET', 'POST'])
def results():
    ##this makes sure that the logged in users only have access to it
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        return render_template('results.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

                # Load the model
                model = pickle.load(open('OVR.pkl','rb'))

                df2 = pd.read_csv('./upload/test2.csv', sep=r'\s*(?:\||\#|\,)\s*',engine='python')
                df2 = df2.drop('Unnamed: 0', axis=1).values
                df2 = np.asarray(df2)

                with open('KNN.pkl','rb') as file:
                    mp = pickle.load(file)

                ##predict
                df2 = mp.predict(df2)

                return render_template('results.html',predict= predict,df2=df2)
                
        elif request.method == 'GET':
            return render_template('predictOVR.html')

###################Predict DNN#####################

@app.route('/predictDNN/')
def predictDNN():
    if not session.get('log'):
        return redirect(url_for('login'))
    else:
        return render_template('predictDNN.html')

@app.route('/predictDNN', methods=['GET', 'POST'])
def upload_file3():
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

                #read the data
                df = pd.read_csv('./upload/joinedData.csv', sep=r'\s*(?:\||\#|\,)\s*',
                                 engine='python')

                #change the classes to numbers
                Class = {'LUAD': 0,'BRCA': 1,'KIRC': 2,'PRAD': 3,'COAD': 4} 
                df.Class = [Class[item] for item in df.Class] 
                df = df.drop('Unnamed: 0',1)
                df = df.drop('Unnamed: 0.1',1)
                df

                #separate the data into X and y
                X = df.drop('Class', axis=1).values
                y = df['Class'].values
                y = np.asarray(y)

                #standardize the data
                X = (X - X.mean()) / (X.max() - X.min())

                #Split the data set to test and train
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
                X_train

                #get the estimator
                estimator = tf.estimator.DNNClassifier(
                    feature_columns=[tf.feature_column.numeric_column('x', shape=X_train.shape[1:])],
                    hidden_units=[1000, 500, 250], 
                    optimizer=tf.train.ProximalAdagradOptimizer(
                      learning_rate=0.01,
                      l1_regularization_strength=0.001
                    ), #optimizer was used to improve the estimator
                    n_classes=5) #the number of label classes, we have 5

                # defining the training inputs
                train = tf.estimator.inputs.numpy_input_fn(
                    x={"x": X_train},
                    y=y_train,
                    batch_size=X_test.shape[0],
                    num_epochs=None,
                    shuffle=False,
                    num_threads=1
                    ) 


                estimator.train(input_fn = train,steps=1000)

                # defining the test inputs
                input_fn2 = tf.estimator.inputs.numpy_input_fn(
                    x={"x": X_test},
                    y=y_test, 
                    shuffle=False,
                    batch_size=X_test.shape[0],
                    num_epochs=None)

                #evaluate the estimator
                estimator.evaluate(input_fn2,steps=1000) 

                #predict input
                df2 = pd.read_csv('./upload/test.csv', sep=r'\s*(?:\||\#|\,)\s*',engine='python')
                df2 = df2.drop('Unnamed: 0', axis=1).values
                df2 = np.asarray(df2)

                pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": df2},
                    shuffle=False)

                #predict the results
                pred_results = estimator.predict(input_fn=pred_input_fn)
               
                return render_template('results.html',predict= predict,pred_results=pred_results)
                
        elif request.method == 'GET':
            return render_template('predictDNN.html')

######################################Predict DNN end########################################

##Route for list route
@app.route('/list')
def list():
    # Selects patients in the database and prints it in the list page
    result = engine2.execute("select * from patient")
    return render_template('list.html', result = result)

####### CHAT FEATURE ########
@socketio.on('enter', namespace='/chat')
def enter(message):
    ##When a user enters the room, then the people in the room will be notified
    room = session.get('room')
    join_room(room)
    emit('status', {'msg': session.get('name')}, room=room)

@socketio.on('leave', namespace='/chat')
def leave(message):
    ##When a user leaves the room, then the people in the room will be notified
    room = session.get('room')
    leave_room(room)
    emit('status', {'msg': session.get('name')}, room=room)

@socketio.on('text', namespace='/chat')
def text(message):
    ##when the user inputted something in the chat and sent, then the other users will be able to see it
    room = session.get('room')
    emit('message', {'msg': session.get('name') + ':' + message['msg']}, room=room)

##Name and room is needed when entering the room
class LoginForm(FlaskForm):
    name = StringField('Name', validators=[Required()])
    room = StringField('Room', validators=[Required()])
    submit = SubmitField('Enter Consultation')

##Where the chat index is, where the user will put the name and the room 
@app.route('/index', methods=['GET', 'POST'])
def index():
    FlaskForm = LoginForm()
    if FlaskForm.validate_on_submit():
        ##Get the name and the room
        session['name'] = FlaskForm.name.data
        session['room'] = FlaskForm.room.data
        ##redirected to chat
        return redirect(url_for('.chat'))
    elif request.method == 'GET':
        FlaskForm.name.data = session.get('name', '')
        FlaskForm.room.data = session.get('room', '')
    return render_template('index.html', FlaskForm=FlaskForm)

##Chat route 
@app.route('/chat')
def chat():
    name = session.get('name', '')
    room = session.get('room', '')
    ##if name and room is blank then it will be redirected to the chat index
    if name == '' or room == '':
        return redirect(url_for('.index'))
    ##else it will be redirected to the chat page
    return render_template('chat.html', name=name, room=room)

######## END CHAT FEATURE######

if __name__ == '__main__':
    ##run the app in port 5000 with local host
    socketio.run(app,debug = False, host='0.0.0.0',port=5000)
