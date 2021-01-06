import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import normalize

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    Age=float(request.form['Age'])
    CigsPerDay=float(request.form['CigsPerDay'])
    Cholestrol=float(request.form['Cholestrol'])
    SysBP=float(request.form['SysBP'])
    DIaBP=float(request.form['DIaBP'])
    BMI=float(request.form['BMI'])
    HeartRate=float(request.form['HeartRate'])
    GlucoseLevel=float(request.form['GlucoseLevel'])
    Gender=float(request.form['Gender'])
    BpMedication=float(request.form['BpMedication'])
    PrevalentStroke=float(request.form['PrevalentStroke'])
    Smoker=float(request.form['Smoker'])
    list_to_be_normalised=np.array([ Age,CigsPerDay, Cholestrol, SysBP,DIaBP, BMI,HeartRate,GlucoseLevel]).reshape(1,-1)
    normalized = normalize(list_to_be_normalised)
    boolean = [Gender,BpMedication,PrevalentStroke,Smoker]
    final_features = np.append(normalized,boolean).reshape(1, -1)
    print(final_features)
    prediction = model.predict(final_features)

    if prediction == 1:
        return render_template('problem.html')
    else:
        return render_template('healthy.html')
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)