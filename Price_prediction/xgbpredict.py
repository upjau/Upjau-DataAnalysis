from flask import Flask,request,jsonify
#from tensorflow.python.keras.models import load_model
import pickle
from sklearn.externals import joblib
import traceback
import numpy as np 
import pandas as pd 

app = Flask(__name__)





@app.route('/predict',methods=['POST'])
def predict():
	if xgb:
		print("Prior Hello")
		try:
			print("Hello0")
			json = request.json
			print("Hello1")
			print(json)
			print("Hello2")
			query = pd.DataFrame(json)
			print("Hello3)")
			query=query.reindex(columns=model_columns,fill_value=0)
			print(query)
			query=np.array(query)
			print(query)

			prediction = xgb.predict(query)

			return jsonify({'prediction':str(prediction)})
		except:
			return jsonify({'trace':traceback.format_exc()})

	else:
		print('Train model first')
		return('no model')
	



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    xgb= joblib.load("xgbmodelfakecl.pickle") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("columns.pickle")
    print(model_columns) # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)