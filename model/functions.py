#Importing required packages.
import pandas as pd
from os.path import exists
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import joblib
import os
from csv import writer
from model.model import Wine

#si wine n'existe pas on abandonne
def train_model():
  """Fonction pour train le model sur les données

  Returns:
      model SVC entrainé si ok
      -1 si resources/Wines.csv n'existe pas
  """
  wine_file_exists = exists('./resources/Wines.csv')
  if wine_file_exists == False:
    return -1
  wine = pd.read_csv('./resources/Wines.csv') #faire un check que le wine existe
  wine = wine.drop('Id', axis = 1)
  # Making binary classificaion for the response variable.
  # Dividing wine as good and bad by giving the limit for the quality
  bins = (1, 5.5, 10)
  group_names = ['bad', 'good']
  wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
  #Now lets assign a labels to our quality variable
  label_quality = LabelEncoder()
  #Bad becomes 0 and good becomes 1 
  wine['quality'] = label_quality.fit_transform(wine['quality'])
  #Now seperate the dataset as response variable and feature variabes
  X = wine.drop('quality', axis = 1)
  y = wine['quality']
  #Train and Test splitting of data 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 14)
  #Applying Standard scaling to get optimized result
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.fit_transform(X_test)
  #model creation
  #Let's run our SVC again with the best parameters.
  svc2 = SVC(C = 1.2, gamma =  1.4, kernel= 'rbf')
  svc2.fit(X_train, y_train)
  return svc2

#on supprime le modele serialise pour ne jamais avoir de difference entre modele et modele serialize
def save_model(model):
  """Fonction pour enregistrer le model dans un fichier

  Args:
      model : le model a enregistrer
  """
  serialized_file_exists = exists('./resources/serialized_model.z')
  if serialized_file_exists == True:
    os.remove('./resources/serialized_model.z') 
  pickle.dump(model, open('./resources/model.sav', 'wb'))

def predict(data):
  """predit le resultat (bad / good) d'un vin

  Args:
      data : le vin a predict

  Returns:
      la prediction du model (bad = 0 / good = 1)
  """
  model_file_exists = exists('./resources/model.sav')
  if model_file_exists == False:
      save_model(train_model(),'./resources/model.sav')
  model = pickle.load(open('./resources/model.sav', 'rb'))
  return model.predict([list(data.values())])

def get_model_metrics(): 
  """renvoie toutes les metrics

  Returns:
      les metrics du model
  """
  model_file_exists = exists('./resources/model.sav')
  if model_file_exists == False:
    save_model(train_model(),'./resources/model.sav')
  model = pickle.load(open('./resources/model.sav', 'rb'))
  return model.get_params()

def serialize_model():
  """on supprime serialized_model.z


  Returns:
      la version serialized du model
  """
  serialized_file_exists = exists('./resources/serialized_model.z')
  if serialized_file_exists == True:
    os.remove('./resources/serialized_model.z') 
  model = pickle.load(open('./resources/model.sav', 'rb'))
  return joblib.dump(model, './resources/serialized_model.z')

def insertWine(w, quality: int):
  """ajout un nouveau vin

  Args:
      w (Wine): le nouveau vin a ajouter
      quality (int): la qualité du nouveau vin

  Returns:
      -1 si le fichiers Wines.csv n'existe pas
  """
  wine_file_exists = exists('./resources/Wines.csv')
  if wine_file_exists == False:
    return -1
  with open('./resources/Wines.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow([w.fixed_acidity,w.volatile_acidity,w.citric_acidity,w.residual_sugar, w.chlorides,w.free_sulfur_dioxide,w.total_sulfur_dioxide,w.density,w.ph,w.sulphates, w.alcohol,quality,w.id])
    f_object.close()

def get_perfect_wine():
  """permet d'obtenir le meilleur vin des données basé sur notre analyse

  Returns:
      Wine: les informations du meilleur vin
  """
  thereIsNoPerfectWine = False
  if thereIsNoPerfectWine:
    return -1
  wine = pd.read_csv('./resources/Wines.csv')
  wine = wine.drop('Id', axis = 1)

  best_wines = wine[wine['alcohol'] == wine['alcohol'].max()]
  best_wines = best_wines[best_wines['volatile acidity'] == best_wines['volatile acidity'].min()]
  perfectWine = best_wines.to_dict('records')[0]
  return perfectWine