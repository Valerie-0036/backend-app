from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from waitress import serve
from pyngrok import ngrok
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
# import tensorflow as tf
# import librosa
from google.cloud.firestore_v1.base_query import FieldFilter,Or
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask_cors import CORS

cred = credentials.Certificate(r"bansos-2016-firebase-adminsdk-max5r-660f490c5d.json")
firebase_admin.initialize_app(cred)
db=firestore.client()
app = Flask(__name__)
# CORS(app)

# app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

@app.route("/api", methods=['POST','GET'])
def home():
  print(request.args['query'])
  input = str(request.args['query'])
  print(input)
  
  answer = get_document("predictions",input)
  X_data=np.array([answer["penghasilan"],answer["asset_motor"],answer["asset_mobil"],answer["asset_rumah"],answer["harga_rumah"],answer["harga_sewa"],answer["jumlah_anak"],answer["tanggungan_lain"]])
  
  datasets = pd.read_csv('data_warga_baruuuuu.csv', sep = ';')
  datasets.keys()
  X = np.asarray(datasets)
  print(X)
  kmeans = KMeans(n_clusters = 2,random_state=0)
  kmeans.fit(X)
  print(kmeans.cluster_centers_)
  #clustered
  print(kmeans.labels_)
  new_y =[]
  for i in range(len(kmeans.labels_)):
    if(kmeans.labels_[i] == 0):
      new_y.append("Layak")
    else:
      new_y.append("Tidak Layak")
  # Import necessary modules



  # Create feature and target arrays
  y = kmeans.labels_

  # Split into training and test set
  X_train, X_test, y_train, y_test = train_test_split(
        X, new_y, test_size = 0.2, random_state=0)

  # neighbors = np.arange(1, 9)
  # train_accuracy = np.empty(len(neighbors))
  # test_accuracy = np.empty(len(neighbors))

  # Loop over K values
  # for i, k in enumerate(neighbors):
  #   knn = KNeighborsClassifier(n_neighbors=k)
  #   knn.fit(X_train, y_train)
  #   predictions=knn.predict(X_test)
  #   print(classification_report(y_test, predictions,digits=4))
  #   print(confusion_matrix(y_test,predictions))


  #kalo udah ada data

  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(X_train, y_train)
  predictions=knn.predict(X_data.reshape(1,8))
  print(predictions)
  update_existing_document(input,predictions[0])
  return predictions[0]

# home()

def update_existing_document(docID,change):

    # Get the reference to the collection

    collection_ref = db.collection("predictions")


    # Get the document you want to update by its ID

    doc_ref = collection_ref.document(docID)


    # Update the document

    doc_ref.update({

        "hasil": change,

    })



    # Get the document you want to update by its ID

    #doc_ref = collection_ref.document('your_document_id')


    # Update the document

    #doc_ref.update({

    #    'field_to_update': 'new_value'

    #})

def get_all_docs(collectionName):

    # Get the reference to the collection

    #collection_ref = db.collection(collectionName)


    docs = (

            db.collection(collectionName)

            .stream()

        )


    # Iterate over the documents and store their IDs and data in a list

    documents_list = []

    for doc in docs:

        doc_data = doc.to_dict()

        doc_data['id'] = doc.id

        doc_data['docData'] = doc._data

        #print(doc._data)

        documents_list.append(doc_data)


    # Print the list of documents

    for doc_data in documents_list:

        print(f"Document ID: {doc_data['id']}")

        print(f"Document Data: {doc_data['docData']}")

        print()


def get_document(collection_name, document_id):

    doc_ref = db.collection(collection_name).document(document_id)

    print(doc_ref)

    doc = doc_ref.get()

    print(doc)

    if doc.exists:

        return doc.to_dict()

    else:

        print(f"Document '{document_id}' not found in collection '{collection_name}'.")

        return None

   

def delete_document(collection_name, document_id):

    try:

        doc_ref = db.collection(collection_name).document(document_id)

        doc_ref.delete()

        print(f"Document with ID {document_id} deleted successfully.")

    except Exception as e:

        print(f"Error deleting document: {str(e)}")


def get_documents_with_status(collection_name, status_value):

    try:

        doc_ref = db.collection(collection_name)

       

        #make your query

        query = doc_ref.where(filter=FieldFilter("status", "==", status_value))


        #stream for results

        docs = query.stream()
       
        for doc in docs:

            data = doc.to_dict()

            print("Document data:", data)

    except Exception as e:

        print(f"Error retrieving documents: {str(e)}")

def get_different_status(collection_name, status_value1, status_value2):

    try:

        doc_ref = db.collection(collection_name)

        filter_todo = FieldFilter("status", "==", status_value1)

        filter_done = FieldFilter("status", "==", status_value2)


        # Create the union filter of the two filters (queries)

        or_filter = Or(filters=[filter_todo, filter_done])


        # Execute the query

        docs = doc_ref.where(filter=or_filter).stream()

        for doc in docs:

            data = doc.to_dict()

            print("Document data:", data)

    except Exception as e:

        print(f"Error retrieving documents: {str(e)}")
        

# model = tf.keras.models.load_model('s_model')
# d = {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark', 4: 'drilling', 5: 'engine_idling', 6:'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music'}

# def func(filename):
#     audio, sample_rate = librosa.load(filename)
#     mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
#     mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
#     predicted_label=np.argmax(model.predict(mfccs_scaled_features),axis=1)
#     return d[predicted_label[0]]

# @app.route('/predict',methods=['POST'])
# def predict():
#     if 'audio' not in request.files:
#         return 'No file provided', 400

#     audio_file = request.files['audio']
#     if not audio_file.filename.lower().endswith('.wav'):
#         return 'Invalid file type, must be .wav', 400
#     preditction = func(audio_file)
#     print(preditction)
#     return preditction

# nama="B"
# usia='30'
# penghasilan=500000
# asset_motor=0
# asset_mobil=0
# asset_rumah=1
# harga_rumah=300000000
# harga_sewa=0
# jumlah_anak=0
# tanggungan_lain=0
# hasil="Tidak Layak Menerima"

# data = {
# 'nama': nama,
# 'usia': usia,
# 'penghasilan': penghasilan,
# 'asset_motor': asset_motor,
# 'asset_mobil': asset_mobil,
# 'asset_rumah': asset_rumah,
# 'harga_rumah': harga_rumah,
# 'harga_sewa' : harga_sewa,
# 'jumlah_anak': jumlah_anak,
# 'tanggungan_lain': tanggungan_lain,
# 'hasil': hasil 
# }

# doc_ref=db.collection('predictions').document()
# doc_ref.set(data)

if __name__ == '__main__':
    # serve(app, host="localhost", port=55895)

    # public_url = ngrok.connect(name='flask').public_url
    # print(" * ngrok URL: " + public_url + " *")
    app.run()
    # serve(app, host="localhost", port=5000)



"""# Example code for setting up a Flask API for audio classification
from flask import Flask, request, jsonify
from tensorflow import keras
import librosa
import numpy as np

app = Flask(__name__)

# Define the endpoint for audio file classification
@app.route('/classify_audio', methods=['POST'])
def classify_audio():
    # Retrieve uploaded audio file from Flutter
    audio_file = request.files['audio'].read()

    # Load the trained deep learning model
    model = keras.models.load_model('audio_model.h5')

    # Preprocess the audio file
    # For example, you can use librosa library to convert the audio file to spectrogram
    spectrogram = librosa.stft(np.frombuffer(audio_file, dtype=np.int16))

    # Make predictions
    predictions = model.predict(spectrogram)

    # Format and send response back to Flutter
    response = {'predictions': predictions.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

"""