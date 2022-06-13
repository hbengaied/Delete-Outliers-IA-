import pandas as pd
import numpy as np
import random as rand
import statistics as stat

import matplotlib.pyplot as plt
import seaborn as sns 

data_df = pd.read_csv("data.csv", sep = '\t')
#raw data from the csv file

#Dans le fichier l'entraînement est fait en même temps que la détermination sur l'ensemble fourni.
#Il n'y a pas (encore ?) de fonction ou l'on fournit une valeur et on a un retour sur si il s'agit d'un outlier ou non

#Leaf class that we use
class DecisionLeaf:
	def __init__(self, size, id_attr = None, std = -1):
		self.tab_size = size
		
		self.outlier_a = [None] * (self.tab_size)
		self.inlier = [None] * (self.tab_size)
		self.outlier_b = [None] * (self.tab_size)
		
		self.id_attr = id_attr
		self.std = std
		self.seuilA = 0
		self.seuilB = 0


	def after_kmean(self, outlier_a, inlier, outlier_b, seuilA, seuilB):
		self.seuilA = seuilA
		self.seuilB = seuilB
		self.outlier_a = outlier_a[:]
		self.inlier = inlier[:]
		self.outlier_b = outlier_b[:]
		

	def find_attr_discriminant(self, data):
		nb_attr = data[0].size
		std = -1;
		selected_attr = -1;
		
		for k in range (nb_attr):
			tmp = np.std(data[:,k])
			#print("tmp = " + str(tmp))
			if (tmp > std):
				std = tmp
				selected_attr = k
		
		print("The selected attribut is " + str(selected_attr) + " with a standard variation of " + str(std))
		self.id_attr = selected_attr
		self.std = std
		return 0

	def kmean(self, data):
		if (self.id_attr == -1):
			print("Il faut d'abord trouver l'attribut discriminant")
			return -1
		Data = data[self.id_attr]
		#print(Data)
		#on initialise les centres
		#assez barbare
		centers = [None] * (3)
		temp = rand.sample(Data, 1)
		centers[0] = temp
		temp = rand.sample(Data, 1)
		while (temp == centers[0]):
			temp = rand.sample(Data, 1)
		centers[1] = temp
		temp = rand.sample(Data, 1)
		while (temp == centers[0] or temp == centers[1]):
			temp = rand.sample(Data, 1)
		centers[2] = temp
		
		#sort of centers
		for i in range(3):
			for j in range(3):
				if (centers[j] > centers[i]):
					temp = centers[i]
					centers[i] = centers[j]
					centers[j] = temp
		
		#on fait un entraînement sur size itérations (devrait être correct)
		#lists = [[],[],[]]
		for k in range(len(Data)):
			lists = [[],[],[]]
			for value in Data:
				dist0 = abs(centers[0][0] - value)
				dist1 = abs(centers[1][0] - value)
				dist2 = abs(centers[2][0] - value)
				
				indice = 0
				#barbare aussi mais c'est sûr
				if (dist0 < dist1):
					if (dist0 < dist2):
						indice = 0
				if (dist1 < dist0):
					if (dist1 < dist2):
						indice = 1
				if (dist2 < dist0):
					if (dist2 < dist1):
						indice = 2
				
				lists[indice].append(value)
				#print(value)
			
			centers[0][0] = stat.mean(lists[0])
			centers[1][0] = stat.mean(lists[1])
			centers[2][0] = stat.mean(lists[2])
			#print("center[0]=" + str(centers[0][0]) + "center[1]=" + str(centers[1][0]) + "center[2]=" + str(centers[2][0]))
			
		#maintenant on remplis la structure	
		self.after_kmean(lists[0], lists[1], lists[2], (max(lists[0])+min(lists[1]))/2, (max(lists[1])+min(lists[2]))/2)
		
		#print(self.inlier)
		return 0



#transforming datagram into numpy array
data = np.array(data_df)
#removing last column which indicates the belonging or not to the outlier group
data = np.delete(data, 2, 1)
leaf = DecisionLeaf(int(data.size / data[0].size))
leaf.find_attr_discriminant(data)

#we need lists for our kmean function
size = np.delete(data, 1, 1).size
x = np.resize(np.delete(data, 1, 1), (1, size))[0]
y = np.resize(np.delete(data, 0, 1), (1, size))[0]
a = [x.tolist(), y.tolist()]
leaf.kmean(a)


#we verify what we have in return
data = np.array(data_df)

nb = 0
TN = 0
FN = 0
TP = 0
FP = 0
#print(int(data.size/data[0].size))
for k in range(int(data.size/data[0].size)):
	if (data[k][2] == 0):
		nb += 1
	if (data[k][0] in leaf.inlier):
		if (data[k][2] == 0) :
			TN += 1
		else :
			FN += 1
		data[k][2] = 0
	else:
		if (data[k][2] == 1) :
			TP += 1
		else :
			FP += 1
		data[k][2] = 1

print("seuilA = " + str(leaf.seuilA))
print("seuilB = " + str(leaf.seuilB))


confusion_matrix = np.array([[TN, FP], [FN, TP]])

#print("Difference of sets :")
#print(data)
print("\nWe have " + str(len(leaf.inlier)) + " inliers with the Decision Leaf and kmean coupled algorithms when there is really " + str(nb))

print("\nReal confusion Matrix:")
print(confusion_matrix)

exact = (TN+TP) / (TN+FP+FN+TP)
exactp = ((TN/(TN+FP)) + (TP/(FN+TP))) / 2
precision = TP / (TP+FP)
rappel = TP / (FN+TP)
print("\nexactitude = " + str(exact))
print("exactitude pondérée = " + str(exactp))
print("precision = " + str(precision))
print("rappel = " + str(rappel))


X=[]
Y=[]
Center=[]
for row in data:
	X.append(float(row[0]))
	Y.append(float(row[1]))
	Center.append(int(row[2]))

for c, x, y in zip(Center, X, Y):
	if (c == 0):
		plt.plot(x, y, color = 'red', marker='o')
	else:
		plt.plot(x, y, color = 'blue', marker='o')
   
plt.xlabel('x')
plt.ylabel('y')
plt.show()




