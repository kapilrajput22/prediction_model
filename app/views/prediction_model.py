import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score #Import scikit-learn metrics module for accuracy calculation# Load dataset
import pickle
url = "/home/kapil.rajput/Downloads/adult_data.csv"

df = pd.read_csv(url)
print(df.columns)
df = df[['age', 'workclass','education','marital-status', 'occupation', 'relationship', 'race', 'sex',
          'capital-gain', 'Capital-loss', 'hours-per-week', 'native-country','class']]
# filling missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", np.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse',
            'Married-civ-spouse', 'Married-spouse-absent',
            'Never-married','Separated','Widowed'],
           ['divorced','married','married','married',
            'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'sex', 'native-country', 'class']
labelEncoder = preprocessing.LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

print(df)
#droping redundant columns
X = df.values[:, 0:12]
Y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                     max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)
print(X_test)
print(y_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini)*100 )

#creating and training a model
#serializing our model to a file called model.pkl
pickle.dump(dt_clf_gini, open("model.pkl","wb"))
def ValuePredictor(to_predict_list):
    dict_updated = {'?': 0,'Federal-gov': 1,'Local-gov': 2,'Never-worked': 3,'Private': 4,'Self-emp-inc': 5,'Self-emp-not-inc': 6,'State-gov': 7,'Without-pay': 8,'Amer-Indian-Eskimo': 0,'Asian-Pac-Islander': 1,'Black': 2,'Other': 3,'White': 4,'10th': 0,'11th': 1,'12th': 2,'1st-4th': 3,'5th-6th': 4,'7th-8th': 5,'9th': 6,'Assoc-acdm': 7,'Assoc-voc': 8,'Bachelors': 9,'Doctorate': 10,'HS-grad': 11,'Masters': 12,'Preschool': 13,'Prof-school': 14,'Some-college': 15,'divorced': 0,'married': 1,'not married': 2,'Married-spouse-absent': 3,'Never-married': 4,'Separated': 5,'Widowed': 6,'Adm-clerical': 1,'Armed-Forces': 2,'Craft-repair': 3,'Exec-managerial': 4,'Farming-fishing': 5,'Handlers-cleaners': 6,'Machine-op-inspct': 7,'Other-service': 8,'Priv-house-serv': 9,'Prof-specialty': 10,'Protective-serv': 11,'Sales': 12,'Tech-support': 13,'Transport-moving': 14,'Husband': 0,'Not-in-family': 1,'Other-relative': 2,'Own-child': 3,'Unmarried': 4,'Wife': 5,'Female': 0,'Male': 1,'Cambodia': 1,'Canada': 2,'China': 3,'Columbia': 4,'Cuba': 5,'Dominican-Republic': 6,'Ecuador': 7,'El-Salvador': 8,'England': 9,'France': 10,'Germany': 11,'Greece': 12,'Guatemala': 13,'Haiti': 14,'Holand-Netherlands': 15,'Honduras': 16,'Hong': 17,'Hungary': 18,'India': 19,'Iran': 20,'Ireland': 21,'Italy': 22,'Jamaica': 23,'Japan': 24,'Laos': 25,'Mexico': 26,'Nicaragua': 27,'Outlying-US(Guam-USVI-etc)': 28,'Peru': 29,'Philippines': 30,'Poland': 31,'Portugal': 32,'Puerto-Rico': 33,'Scotland': 34,'South': 35,'Taiwan': 36,'Thailand': 37,'Trinadad&Tobago': 38,'United-States': 39,'Vietnam': 40,'Yugoslavia': 41,'<=50K': 0,'>50K': 1}
    to_predict_list_updated = [dict_updated.get(item,item)  for item in to_predict_list]
    print(to_predict_list_updated)
    to_predict = np.array(to_predict_list_updated).reshape(1,12)
    print(to_predict)
    import os
    path = os.path.join(os.getcwd(), "model.pkl")
    print(path)

    try:
        loaded_model = pickle.load(open(path,"rb"))
    except:
        print("No Data Found")
    result = loaded_model.predict(to_predict)[0]
    print(result)
    if result==1:
        prediction='Income more than 50K'
    else:
        prediction='Income less that 50K'

    return prediction


to_predict_list=['20','Federal-gov','10th','divorced','Adm-clerical','Husband','Black','Female','12345','1234','24','India']
result = ValuePredictor(to_predict_list)
print(result)

