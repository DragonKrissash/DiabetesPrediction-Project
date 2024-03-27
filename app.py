from flask import Flask,request,render_template
import pickle as pkl
import numpy as np
import pandas as pd

app=Flask(__name__)

scaler=pkl.load(open('models/scaler.pkl','rb'))
reg=pkl.load(open('models/logistic_regressor.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        preg=float(request.form.get('Pregnancies'))
        glu=float(request.form.get('Glucose'))
        bp=float(request.form.get('BloodPressure'))
        st=float(request.form.get('SkinThickness'))
        ins=float(request.form.get('Insulin'))
        bmi=float(request.form.get('BMI'))
        dpf=float(request.form.get('DiabetesPedigreeFunction'))
        age=float(request.form.get('Age'))
        data=scaler.transform([[preg,glu,bp,st,ins,bmi,dpf,age]])
        pred=reg.predict(data)
        if pred[0]==1:
            result='Diabetic'
        else:
            result='Non-Diabetic'
        return render_template('home.html',result=result)
    else:
        return render_template('home.html')


if __name__=='__main__':
    app.run(host='0.0.0.0')