import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def calcul_nb_donnee():
	data_df = pd.read_csv("data.csv")
	df_columns = data_df.columns.values.tolist()
	features = df_columns[0:3] # 3 premieres colonnes
	X = data_df[features]
	test = np.array(X)
	#print(test)
	print(int(test.size / test[0].size)+1)


def data_visualtion():
	data = np.array(pd.read_csv("data.csv", delimiter='\t'))
	X=[]
	Y=[]
	Center=[]
	for row in data:
		X.append(float(row[0]))
		Y.append(float(row[1]))
		Center.append(int(row[2]))
	#plt.scatter(X, Y)

	for c, x, y in zip(Center, X, Y):
		if (c == 0):
			plt.plot(x, y, color = 'red', marker='o')
		else:
			plt.plot(x, y, color = 'blue', marker='o')
    
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
    


calcul_nb_donnee()
data_visualtion()






