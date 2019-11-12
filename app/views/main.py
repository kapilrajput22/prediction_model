from flask import render_template, jsonify, Flask, redirect, url_for, request
from app import app
import random
import os
import numpy as np
import sklearn
import pickle

@app.route('/')

#disease_list = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', \
                  # 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', \
                  # 'Hernia']


def ValuePredictor(to_predict_list):
    dict_updated = {'?': 0,'Federal-gov': 1,'Local-gov': 2,'Never-worked': 3,'Private': 4,'Self-emp-inc': 5,'Self-emp-not-inc': 6,'State-gov': 7,'Without-pay': 8,'Amer-Indian-Eskimo': 0,'Asian-Pac-Islander': 1,'Black': 2,'Other': 3,'White': 4,'10th': 0,'11th': 1,'12th': 2,'1st-4th': 3,'5th-6th': 4,'7th-8th': 5,'9th': 6,'Assoc-acdm': 7,'Assoc-voc': 8,'Bachelors': 9,'Doctorate': 10,'HS-grad': 11,'Masters': 12,'Preschool': 13,'Prof-school': 14,'Some-college': 15,'divorced': 0,'married': 1,'not married': 2,'Married-spouse-absent': 3,'Never-married': 4,'Separated': 5,'Widowed': 6,'Adm-clerical': 1,'Armed-Forces': 2,'Craft-repair': 3,'Exec-managerial': 4,'Farming-fishing': 5,'Handlers-cleaners': 6,'Machine-op-inspct': 7,'Other-service': 8,'Priv-house-serv': 9,'Prof-specialty': 10,'Protective-serv': 11,'Sales': 12,'Tech-support': 13,'Transport-moving': 14,'Husband': 0,'Not-in-family': 1,'Other-relative': 2,'Own-child': 3,'Unmarried': 4,'Wife': 5,'Female': 0,'Male': 1,'Cambodia': 1,'Canada': 2,'China': 3,'Columbia': 4,'Cuba': 5,'Dominican-Republic': 6,'Ecuador': 7,'El-Salvador': 8,'England': 9,'France': 10,'Germany': 11,'Greece': 12,'Guatemala': 13,'Haiti': 14,'Holand-Netherlands': 15,'Honduras': 16,'Hong': 17,'Hungary': 18,'India': 19,'Iran': 20,'Ireland': 21,'Italy': 22,'Jamaica': 23,'Japan': 24,'Laos': 25,'Mexico': 26,'Nicaragua': 27,'Outlying-US(Guam-USVI-etc)': 28,'Peru': 29,'Philippines': 30,'Poland': 31,'Portugal': 32,'Puerto-Rico': 33,'Scotland': 34,'South': 35,'Taiwan': 36,'Thailand': 37,'Trinadad&Tobago': 38,'United-States': 39,'Vietnam': 40,'Yugoslavia': 41,'<=50K': 0,'>50K': 1}
    to_predict_list_updated = [dict_updated.get(item,item)  for item in to_predict_list]
    print(to_predict_list_updated)
    to_predict = np.array(to_predict_list_updated).reshape(1,12)
    print(to_predict)
    print(os.getcwd())
    path = os.path.join(os.getcwd(), "model.pkl")
    from pickle import load
    print(path)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)[0]
    print(result)
    if result==1:
        prediction='Income more than 50K'
    else:
        prediction='Income less that 50K'


    print(prediction)
    return prediction



@app.route('/result',methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        print(to_predict_list)
        prediction = ValuePredictor(to_predict_list)
        '''
        to_predict_list = list(map(int, to_predict_list))
        prediction = ValuePredictor(to_predict_list)
        print(prediction)
        '''
    return render_template('result.html', predictions=prediction)


@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')