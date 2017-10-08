from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import base64


import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import * 

app = Flask(__name__)
global classifier, graph
classifier, graph = init()

	
@app.route('/')
def index():
	return render_template("webPage.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	
	convertImage(imgData)
	print ("Image converted ")
    
	x = imread('output.png',mode='L')
	x = np.invert(x)
	x = imresize(x,(28,28))
	x = x.reshape(1,28,28,1)
    
	print ("loaded,reshaped,Inverted")
	with graph.as_default():
		out = classifier.predict(x)
		print("Predictions  : ",out)

		response = np.array_str(np.argmax(out,axis=1))
		return response	
    

		print(np.argmax(out,axis=1))        
		print ("#####debug3#########")




def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))


	

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5001))
	app.run( port=port)
	#app.run(debug=True)
