from flask import Flask , render_template , request , redirect , url_for
from sklearn.neighbors import KNeighborsClassifier
app = Flask(__name__)

_code = ""
@app.route('/')
def index2():
	return render_template('index.html')
@app.route('/Register')
def Register():
	return render_template('register.html')

@app.route("/users/index", methods=["GET","POST"])
def main():
		return render_template('index.html')

@app.route("/users/graph", methods=["GET","POST"])
def login():
		return render_template('chart.html')

@app.route("/users/logs", methods=["GET","POST"])
def logs():
		return render_template('logs.html')


@app.route("/users/analyse", methods=["POST"])
def analyse():
	if(request.method == "POST"):
		import pandas as pd
		import numpy as np
		# import os
		optimum = pd.read_excel(r"F:\Projects\CropFull\optimum2.xlsx", 'newData')
		price = pd.read_excel(r"F:\Projects\CropFull\optimum2.xlsx", 'pricePerhr')
		optimum['N'] = optimum.N.astype(float)
		optimum['P'] = optimum.P.astype(float)
		optimum['K'] = optimum.K.astype(float)
		optimum['TEMPERATURE'] = optimum.TEMPERATURE.astype(float)
		X = optimum.drop("CLASS",axis=1)
		y = optimum.CLASS

		from sklearn.neighbors import KNeighborsClassifier
		clf = KNeighborsClassifier(n_neighbors=3)
		clf.fit(X,y)

		print(request.form.get('Potassium'))
		
		if(request.form.get('Potassium') == None):
			
			pred = pd.read_excel(r'F:\Projects\CropFull\optimum2.xlsx', 'Sheet3')
			
			prediction = clf.predict(pred)
			print(prediction)

			optimum = optimum[optimum['CLASS'] != prediction[0]]
			X = optimum.drop("CLASS",axis=1)
			y = optimum.CLASS
			clf = KNeighborsClassifier(n_neighbors=3)
			clf.fit(X,y)
			prediction1 = clf.predict(pred)
			print(prediction1)


			optimum = optimum[optimum['CLASS'] != prediction1[0]]
			X = optimum.drop("CLASS",axis=1)
			y = optimum.CLASS
			clf = KNeighborsClassifier(n_neighbors=3)
			clf.fit(X,y)
			prediction2 = clf.predict(pred)
			print(prediction2)
			p1 = prediction1[0]
			p2 = prediction2[0]
			p1 = p1 -1
			p2 = p2 -1
			# print()

			if(prediction == 7):
				return render_template('crops.html' , crop="TOMATO" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[6] ,price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2]) 
			elif(prediction == 1):
				return render_template('crops.html' , crop="GARLIC" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[[0]] ,price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 2):
				return render_template('crops.html' , crop="ONION" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[[1]] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 3):
				return render_template('crops.html' , crop="ORANGE" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[[2]] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 4):
				return render_template('crops.html' , crop="PEAS" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[[3]] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 5):
				return render_template('crops.html' , crop="POTATO" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[[4]] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 6):
				return render_template('crops.html' , crop="RICE" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[[5]] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 8):
				return render_template('crops.html' , crop="SUGARCANE" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[[7]] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])		
			else:
				return "no"

		else:
			potassium = request.form.get('Potassium')
			phosphorous = request.form.get('Phosphorous')
			nitrogen = request.form.get('Nitrogen') 
			pH = request.form.get('pH')
			temperature = request.form.get('Temperature')

			columns = ['N','P','K','pH','TEMPERATURE'] 
			values = np.array([ nitrogen ,phosphorous ,potassium ,   pH , temperature])
			pred = pd.DataFrame(values.reshape(-1, len(values)),columns=columns)
			# print(pred.dtype)
			print(pred)

			prediction = clf.predict(pred)
			print(prediction)

			optimum = optimum[optimum['CLASS'] != prediction[0]]
			X = optimum.drop("CLASS",axis=1)
			y = optimum.CLASS
			clf = KNeighborsClassifier(n_neighbors=3)
			clf.fit(X,y)
			prediction1 = clf.predict(pred)
			print(prediction1)


			optimum = optimum[optimum['CLASS'] != prediction1[0]]
			X = optimum.drop("CLASS",axis=1)
			y = optimum.CLASS
			clf = KNeighborsClassifier(n_neighbors=3)
			clf.fit(X,y)
			prediction2 = clf.predict(pred)
			print(prediction2)

			p1 = prediction1[0]
			p2 = prediction2[0]
			p1 = p1 -1
			p2 = p2 -1
			# print()

			if(prediction == 7):
				return render_template('crops.html' , crop="TOMATO" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[6] ,price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2]) 
			elif(prediction == 1):
				return render_template('crops.html' , crop="GARLIC" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[0] ,price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 2):
				return render_template('crops.html' , crop="ONION" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[1] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 3):
				return render_template('crops.html' , crop="ORANGE" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[2] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 4):
				return render_template('crops.html' , crop="PEAS" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[3] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 5):
				return render_template('crops.html' , crop="POTATO" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[4] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 6):
				return render_template('crops.html' , crop="RICE" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[5] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])
			elif(prediction == 8):
				return render_template('crops.html' , crop="SUGARCANE" , crop1=prediction1[0] , crop2=prediction2[0] , price=price["Price/hr"].iloc[7] , price1=price["Price/hr"].iloc[p1] , price2=price["Price/hr"].iloc[p2])		
			else:
				return "no"
	# render_template('index.html')
	else:
		# return render_template('index.html')
		pass

if (__name__ == "__main__"):	
	app.run()
