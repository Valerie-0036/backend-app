from flask import Flask, request
import numpy as np
import pandas as pd
from waitress import serve
import os
from google.cloud.firestore_v1.base_query import FieldFilter,Or
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask_cors import CORS

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode

cred = credentials.Certificate(r"bansos-2016-firebase-adminsdk-max5r-17c3377da3.json")
firebase_admin.initialize_app(cred)
db=firestore.client()
app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')


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
        


@app.route("/api", methods=['POST','GET'])
def home():
  print(request.args['query'])
  input = str(request.args['query'])
  print(input)
  
  answer = get_document("predictions",input)
  datainput=np.array([answer["penghasilan"],answer["asset_motor"],answer["asset_mobil"],answer["asset_rumah"],answer["harga_rumah"],answer["harga_sewa"],answer["jumlah_anak"],answer["tanggungan_lain"]]).reshape(1,8)
  
  xls = pd.read_excel("Data_KFold.xlsx", sheet_name=['Utama','Layak', 'TidakLayak', 'Latih1','Uji1','Latih2','Uji2','Latih3','Uji3'])
  latih1 = xls['Latih1'].to_numpy()
  uji1 = xls['Uji1'].to_numpy()
  latih2 = xls['Latih2'].to_numpy()
  uji2 = xls['Uji2'].to_numpy()
  latih3 = xls['Latih3'].to_numpy()
  uji3 = xls['Uji3'].to_numpy()
  
  knn1 = KNeighborsClassifier(n_neighbors=3)
  knn1.fit(latih1[:, 0:8], latih1[:, 8:9])
  predictions1 = knn1.predict(uji1[:, 0:8])
  print(classification_report(uji1[:, 8:9], predictions1))
  
  knn2 = KNeighborsClassifier(n_neighbors=3)
  knn2.fit(latih2[:, 0:8], latih2[:, 8:9])
  predictions2 = knn2.predict(uji2[:, 0:8])
  print(classification_report(uji2[:, 8:9], predictions2))
  
  knn3 = KNeighborsClassifier(n_neighbors=3)
  knn3.fit(latih3[:, 0:8], latih3[:, 8:9])
  predictions3 = knn3.predict(uji3[:, 0:8])
  print(classification_report(uji3[:, 8:9], predictions3))
  
  score1 = accuracy_score(uji1[:, 8:9], predictions1)
  score2 = accuracy_score(uji2[:, 8:9], predictions2)
  score3 = accuracy_score(uji3[:, 8:9], predictions3)
  print(score1, score2, score3)
  mean = (score1 + score2 + score3)/3
  print(mean)
  
  predictions1 = knn1.predict(datainput)
  predictions2 = knn2.predict(datainput)
  predictions3 = knn3.predict(datainput)
  hasil = [predictions1[0], predictions2[0], predictions3[0]]

  prediction=mode(hasil)
  update_existing_document(input,prediction)
  return prediction

@app.route("/delete", methods=['POST','GET'])
def delete():
  input = str(request.args['query'])
  print(input)
  delete_document("predictions",input)
  return "Success"
  
  

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