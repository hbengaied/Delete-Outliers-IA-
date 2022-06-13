import pandas as pd
import numpy as np
import random as rand
import statistics as stat

import matplotlib.pyplot as plt
import seaborn as sns 

######## ARGUMENTS !!! ###########
profondeur = 1

##################################


#a changer parfois entre ',' et '\t' cela depends du systeme ou de l'application qui a sauvegarde le fichier
data_df = pd.read_csv("data.csv", sep = '\t')
#raw data from the csv file

data = np.array(data_df)
global_size = int(data.size / data[0].size)

class Leaf:
	def __init__(self, inlier = None, center = False):
		self.type = "leaf"
		self.inlier = inlier
		self.center = center
		

#Node class that we use
class Node:
	def __init__(self, size, id_attr = -1, seuilA = 0, seuilB = 0, L=None, M=None, R=None, std = -1):
		self.type = "node"
		self.tab_size = size
		
		self.outlier_a = [None] * (self.tab_size)
		self.inlier = [None] * (self.tab_size)
		self.outlier_b = [None] * (self.tab_size)
		
		self.L = L
		self.M = M
		self.R = R
		
		self.id_attr = id_attr
		self.rid_attr = -1
		self.std = std
		self.seuilA = seuilA
		self.seuilB = seuilB



	def after_kmean(self, outlier_a, inlier, outlier_b, seuilA, seuilB):
		self.seuilA = seuilA
		self.seuilB = seuilB
		self.outlier_a = outlier_a[:]
		self.inlier = inlier[:]
		self.outlier_b = outlier_b[:]


	
	def find_attr_discriminant(self, data, list_attr, mode = -1):
		#je ne me souviens plus pourquoi j'ai introduit le mode dans cette fonction
		nb_attr = data[0].size
		std = -1;
		selected_attr = -1;
		
		if (mode == -1):
			#le -1 est du au fait qu'ici on n'enleve pas la colonne de vérification 
			for k in range (nb_attr-1):
				tmp = np.std(data[:,k])
				if (tmp > std):
					std = tmp
					selected_attr = k
			
			print("The selected attribut is " + str(list_attr[selected_attr]) + " with a standard variation of " + str(std))
			self.id_attr = selected_attr
			self.rid_attr = list_attr[selected_attr]
			self.std = std
		else:
			print("The selected attribut is " + str(selected_attr) + " with a standard variation of " + str(std))
			self.id_attr = mode
			self.std = np.std(data[k])
		return 0



	def kmean(self, data):
		if (self.id_attr == -1):
			print("Il faut d'abord trouver l'attribut discriminant")
			return -1
		Data = data[:, self.id_attr]
		Data = Data.tolist()
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
			
			centers[0][0] = stat.mean(lists[0])
			centers[1][0] = stat.mean(lists[1])
			centers[2][0] = stat.mean(lists[2])
				
		#maintenant on remplis la structure
		self.after_kmean(lists[0], lists[1], lists[2], (max(lists[0])+min(lists[1]))/2, (max(lists[1])+min(lists[2]))/2)
		
		return 0


def getSplitParameters(data, list_attr):
	if (len(list_attr) == 0):
		print("Error list_attr list is empty")
		return -1, -1, -1
	node = Node(int(data.size/data[0].size))
	node.find_attr_discriminant(data, list_attr)
	D = data[:]
	while(not(node.rid_attr in list_attr)):
		D = np.delete(D, node.id_attr, 1)
		node.find_attr_discriminant(D, list_attr)
	node.kmean(data)
	return node.id_attr, node.seuilA, node.seuilB


def buildDecisionTree(D, central, list_attr, recur):
	real_size = int(D.size/D[0].size)
	liste = list_attr[:]
	if (real_size >= 4 and recur > 0):
		if (len(liste) >= 1):
			current_attr, seuilA, seuilB = getSplitParameters(D, liste)
			Dl = []
			Dm = []
			Dr = []
			
			for value in D:
				temp = value[current_attr]
				if (temp <= seuilA):
					Dl.append(value)
				if (temp > seuilA and temp <= seuilB):
					Dm.append(value)
				if (temp > seuilB):
					Dr.append(value)

			L = buildDecisionTree(np.array(Dl), False, liste, recur-1)
			M = buildDecisionTree(np.array(Dm), True, liste, recur-1)
			R = buildDecisionTree(np.array(Dr), False, liste, recur-1)
			return Node(global_size, current_attr, seuilA, seuilB, L, M, R)
		else:
			return Leaf(D, central)
	else:
		return Leaf(D, central)


#we verify what we have in return
def printDecisionTree(rec=0, n=None, name=""):
	nb = 0; TN = 0; FN = 0; TP = 0; FP = 0
	if (n == None):
		return
	spaces = ""
	for i in range(rec):
		spaces = spaces + "\t"
	
	if (n.type == "node"):
		print(spaces + "node " + str(rec) + " :")
		nb_t, TN_t, FN_t, TP_t, FP_t = printDecisionTree(rec+1, n.L, name + "L")
		nb += nb_t; TN += TN_t; FN += FN_t; TP += TP_t; FP += FP_t
		nb_t, TN_t, FN_t, TP_t, FP_t = printDecisionTree(rec+1, n.M, name + "M")
		nb += nb_t; TN += TN_t; FN += FN_t; TP += TP_t; FP += FP_t
		nb_t, TN_t, FN_t, TP_t, FP_t = printDecisionTree(rec+1, n.R, name + "R")
		nb += nb_t; TN += TN_t; FN += FN_t; TP += TP_t; FP += FP_t
	elif (n.type == "leaf"):
		mode = -1;
		if (n.center is True):
			print(spaces + "leaf " + name + "I :")
			mode = 0;
			#in
		else:
			print(spaces + "leaf " + name + "O :")
			mode = 1;
			#out
		
		for value in n.inlier:
			nb += 1
			if (value[2] == mode):
				if (mode == 0):
					TN += 1
				else:
					TP += 1
			else:
				if (mode == 0):
					FN += 1
				else:
					FP += 1
			print(spaces + "\t" + str(value))
	else:
		print(spaces + "ERROR !")
	
	return nb, TN, FN, TP, FP


#main
node = buildDecisionTree(data, True, [0,1], profondeur)
nb, TN, FN, TP, FP = printDecisionTree(0, node)

#print("seuilA = " + str(node.seuilA))
#print("seuilB = " + str(node.seuilB))

#rest of data verification
confusion_matrix = np.array([[TN, FP], [FN, TP]])

#print("Difference of sets :")
#print(data)
print("\nWe have " + str(nb) + " entries with the Decision Leaf and kmean coupled algorithms when there is really " + str(global_size))

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


def make_row(node):
	X=[]; Y=[]; Center=[]
	if (node.type == "node"):
		X_t, Y_t, C_t = make_row(node.L)
		X.extend(X_t); Y.extend(Y_t); Center.extend(C_t)
		X_t, Y_t, C_t = make_row(node.M)
		X.extend(X_t); Y.extend(Y_t); Center.extend(C_t)
		X_t, Y_t, C_t = make_row(node.R)
		X.extend(X_t); Y.extend(Y_t); Center.extend(C_t)
	else:
		for n in node.inlier:
			X.append(float(n[0]))
			Y.append(float(n[1]))
			if (node.center == True):
				Center.append(0)
			else:
				Center.append(1)
	return X, Y, Center

X, Y, Center = make_row(node)


for c, x, y in zip(Center, X, Y):
	print(c)
	if (c == 0):
		plt.plot(x, y, color = 'red', marker='o')
	else:
		plt.plot(x, y, color = 'blue', marker='o')
   
plt.xlabel('x')
plt.ylabel('y')
plt.show()


