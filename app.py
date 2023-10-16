#pip3 install opencv-contrib-python
from sklearn.preprocessing import LabelEncoder
from flask import Flask,render_template,request,redirect,url_for,flash,session
import numpy as np
import mysql.connector
import cv2,os
import pandas as pd
from PIL import Image
import datetime
import time
import math
import pickle
import requests
import csv


app=Flask(__name__)
app.config['SECRET_KEY']='attendance system'
db = mysql.connector.connect(host="localhost", user="root", passwd="", database="criminal_database")
cur = db.cursor()

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login1')
def login1():
    return render_template('login1.html')

@app.route('/admin_home',methods=["POST","GET"])
def admin_home():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpasword = request.form['userpassword']

        if useremail == 'admin@gmail.com' and userpasword =="admin":
            msg = 'Login Successfull !!!!'
            return render_template('admin_home.html',name=msg)
        else:
            msg = 'Invalid Credentials'
            return render_template('login1.html', name = msg)
    return render_template('login1.html')

@app.route('/adduser',methods=["POST","GET"])
def adduser():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
       
        contact = request.form['contact']
        address = request.form['address']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,contact,address)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,contact,address)
                cur.execute(sql,val)
                db.commit()
                flash(" User Added Successfully","success")
                return render_template("adduser.html")
            else:
                flash("Details already Exists","warning")
                return render_template("adduser.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("adduser.html")
    return render_template('adduser.html')
@app.route('/addvolunteer',methods=["POST","GET"])
def addvolunteer():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        confemail = request.form['confemail']       
        contact = request.form['contact']
        address = request.form['address']
        if useremail == confemail:
            sql="select * from volunteer where Email='%s' and contact='%s'"%(useremail,contact)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into volunteer(Name,Email,Contact,Address)values(%s,%s,%s,%s)"
                val=(username,useremail,contact,address)
                cur.execute(sql,val)
                db.commit()
                flash(" Volunteer  Added Successfully","success")
                return render_template("addvolunteer.html")
            else:
                flash("Details already Exists","warning")
                return render_template("addvolunteer.html")
        else:
            flash("Email doesn't match", "warning")
            return render_template("addvolunteer.html")
    return render_template('addvolunteer.html')

@app.route("/addface", methods=['POST','GET'])
def addface():
    if request.method=='POST':
        Id=request.form['id']
        name=request.form['name']
        crimedetail = request.form['crimedetail']
        print(type(Id))
        print(type(name))
        print(type(crimedetail))
        if not Id:
           flash("Please enter roll number properly ","warning")
           return render_template('addface.html')


        elif not name:
            flash("Please enter your name properly ", "warning")
            return render_template('addface.html')
        elif not crimedetail:
            flash("Please enter mobile number","warning")
        # elif (Id.isalpha() and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "Haarcascade/haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        df = pd.read_csv("Criminal_Details/criminal.csv")
        val = df.Id.values
        if Id in str(val):
            flash("Id already exists", "danger")
            return render_template("addface.html")

        else:
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # incrementing sample number
                    sampleNum = sampleNum + 1
                        # saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage/ " + name + "." + Id + '.' + str(
                            sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                        # display the frame

                else:
                    cv2.imshow('frame', img)
                    # wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                    # break if the sample number is morethan 100
                elif sampleNum > 150:
                        break
        cam.release()
        cv2.destroyAllWindows()
        
        row = [Id, name, crimedetail]
        with open('Criminal_Details/criminal.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        flash("Captured images successfully!!","success")
        return render_template("addface.html")
      
    return render_template("addface.html")
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = str(os.path.split(imagePath)[-1].split(".")[0])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

@app.route('/train')
def train():
    le = LabelEncoder()
    faces, Id = getImagesAndLabels("TrainingImage")
    Id = le.fit_transform(Id)
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"Trained_Model\Trainner.yml")

    flash("Model Trained Successfully", "success")
    return render_template('addface.html')
@app.route('/trackface')
def trackface():
    return render_template('trackface.html')

@app.route("/TrackImages")
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"Trained_Model\Trainner.yml")
    harcascadePath = r"Haarcascade\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    global cam
    df = pd.read_csv("Criminal_Details/criminal.csv")
    cam = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pkl_file = open('label_encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    global tt
    count = []
    flag = 0
    det = 0
    global val_data, global_stop
    global_stop = False
    while True:
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                det += 1
                tt = le.inverse_transform([Id])
                tt=tt[0]
            else:
                tt='Unknown'
                # print(tt)
                # r1 = (tt)
                #
                # if det==50:
                #     sql="select * from volunteer"
                #     x=pd.read_sql_query(sql,db)
                #     pno=x['Contact'].values
                    # for i in pno:
                    #     for j in i:
                    #         import random
                    #         mobile_no=j
                    #         mobile_no=random.ra

                # print(r1)
                # rno = str(tt)



            cv2.putText(frame,str(tt), (x, y + h),font, 1, (255, 255, 255), 2)
           
        cv2.imshow('im', frame)
        if (cv2.waitKey(1) == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()
    return render_template("user_home.html")



@app.route("/TrackImages1")
def TrackImages1():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"Trained_Model\Trainner.yml")
    harcascadePath = r"Haarcascade\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    global cam
    df = pd.read_csv("Criminal_Details/criminal.csv")
    cam = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pkl_file = open('label_encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    global tt
    count = []
    flag = 0
    det = 0
    global val_data, global_stop
    global_stop = False
    while True:
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                det += 1
                tt = le.inverse_transform([Id])
                tt=tt[0]
            else:
                tt='Unknown'
                # print(tt)
                # r1 = (tt)
                #
                # if det==50:
                #     sql="select * from volunteer"
                #     x=pd.read_sql_query(sql,db)
                #     pno=x['Contact'].values
                    # for i in pno:
                    #     for j in i:
                    #         import random
                    #         mobile_no=j
                    #         mobile_no=random.ra

                # print(r1)
                # rno = str(tt)



            cv2.putText(frame,str(tt), (x, y + h),font, 1, (255, 255, 255), 2)
           
        cv2.imshow('im', frame)
        if (cv2.waitKey(1) == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()
    return render_template("trackface.html")


@app.route('/userlogin',methods=['POST','GET'])
def userlogin():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        print(data)
        print(data)

        if data !=[]:
            username = data[0][1]
            session['username']=username
            return render_template("user_home.html",name=username)
        else:
            flash("Account Details doesn't exist",'warning')
            return redirect(url_for('userlogin'))
    return render_template('userlogin.html')

@app.route('/userregistration',methods=["POST","GET"])
def userregistration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
       
        contact = request.form['contact']
        address = request.form['address']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,contact,address)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,contact,address)
                cur.execute(sql,val)
                db.commit()
                flash(" You registered successfully....","success")
                return render_template("userlogin.html")
            else:
                flash("Details already Exists","warning")
                return render_template("userregistration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("userregistration.html")
    return render_template('userregistration.html')






@app.route('/emergency')
def emergency():

    sql="select * from volunteer"
    x=pd.read_sql_query(sql,db)
    pno=x['Contact'].values
    # for i in pno:
    #     for j in i:
    #         import random
    #         mobile_no=j
    #         mobile_no=random.ra
    import random
    # secure random generator
    arr=np.array(pno)
    pno = arr.tolist()
    secure_random = random.SystemRandom()
    pno = secure_random.choice(pno)

    print (pno)

    url = "https://www.fast2sms.com/dev/bulkV2"
    print(url)
    
    message = 'Please Help me'
    no = pno
    data = {
        "route": "q",
        "message": message,
        "language": "english",
        "flash": 0,
        "numbers": no,
    }
    
    headers = {
        "authorization": "9qXilM8snkgvUPYaBDWISdfEO67ZVAtru5GFTmRexhQCL1jpJH2r8GPja9NTuZhQ7wMI5YdgSxOWyAUB",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    return render_template('user_home.html')

@app.route('/emergency1')
def emergency1():

    sql="select * from volunteer"
    x=pd.read_sql_query(sql,db)
    pno=x['Contact'].values
    # for i in pno:
    #     for j in i:
    #         import random
    #         mobile_no=j
    #         mobile_no=random.ra
    import random
    # secure random generator
    arr=np.array(pno)
    pno = arr.tolist()
    secure_random = random.SystemRandom()
    pno = secure_random.choice(pno)

    print (pno)

    url = "https://www.fast2sms.com/dev/bulkV2"
    print(url)
    
    message = 'Please Help me'
    no = pno
    data = {
        "route": "q",
        "message": message,
        "language": "english",
        "flash": 0,
        "numbers": no,
    }
    
    headers = {
        "authorization": "9qXilM8snkgvUPYaBDWISdfEO67ZVAtru5GFTmRexhQCL1jpJH2r8GPja9NTuZhQ7wMI5YdgSxOWyAUB",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    return render_template('trackface.html')

@app.route('/viewvolunteer')
def viewvolunteer():
    sql="select * from volunteer"
    data=pd.read_sql_query(sql,db)
    return render_template('addvolunteer.html',cols=data.columns.values,rows=data.values.tolist())

@app.route('/viewuser')
def viewuser():
    sql="select * from user"
    data=pd.read_sql_query(sql,db)
    return render_template('adduser.html',cols=data.columns.values,rows=data.values.tolist())

if __name__=='__main__':
    app.run(debug=True)