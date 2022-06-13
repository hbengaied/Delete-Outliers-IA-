import pandas as pd
import numpy as np
import random as rand
import statistics as stat

#a changer parfois entre ',' et '\t' cela depends du systeme ou de l'application qui a sauvegarde le fichier
data_df = pd.read_csv("data.csv", sep = '\t')
#raw data from the csv file

data = np.array(data_df)
global_size = int(data.size / data[0].size)

class Leaf:
	def __init__(self, inlier = None):
		self.type = "leaf"
		self.inlier = inlier
		

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
			
			
			#print(list_attr)
			#print (selected_attr)
			#print(data)
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
			#print("center[0]=" + str(centers[0][0]) + " center[1]=" + str(centers[1][0]) + " center[2]=" + str(centers[2][0]))
		
		#maintenant on remplis la structure
		self.after_kmean(lists[0], lists[1], lists[2], (max(lists[0])+min(lists[1]))/2, (max(lists[1])+min(lists[2]))/2)
		
		#print(self.inlier)
		return 0


def getSplitParameters(data, list_attr):
	if (len(list_attr) == 0):
		print("Error list_attr list is empty")
		return -1, -1, -1
	node = Node(global_size)
	node.find_attr_discriminant(data, list_attr)
	D = data[:]
	while(not(node.rid_attr in list_attr)):
		D = np.delete(D, node.id_attr, 1)
		node.find_attr_discriminant(D, list_attr)
	node.kmean(data)
	return node.id_attr, node.seuilA, node.seuilB


def directDecision(outlier):
	print("Je mange des nouilles")
	return 0


def buildDecisionTree(D, central, list_attr, tempAr):
	#print(D)
	liste = list_attr[:]
	if (global_size >= 4):
		if (len(liste) >= 1):
			current_attr, seuilA, seuilB = getSplitParameters(tempAr, liste)
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
			
			#print(liste)
			#print(current_attr)
			liste.pop(current_attr)
			#print(liste)

			L = buildDecisionTree(np.array(Dl), False, liste, np.delete(tempAr,current_attr,1))
			#print(np.array(Dm))
			M = buildDecisionTree(np.array(Dm), True, liste, np.delete(tempAr,current_attr,1))
			R = buildDecisionTree(np.array(Dr), False, liste, np.delete(tempAr,current_attr,1))
			return Node(global_size, current_attr, seuilA, seuilB, L, M, R)
		else:
			return Leaf(D)
	else:
		if (central == True):
			print("coucou")
			return directDecision(outlier=False)
		else:
			print("au revoir")
			return directDecision(outlier=True)


def printDecisionTree(rec=0, n=None, name=""):
	if (n == None):
		return
	spaces = ""
	for i in range(rec):
		spaces = spaces + "\t"
	
	if (n.type == "node"):
		print(spaces + "node " + str(rec) + " :")
		printDecisionTree(rec+1, n.L, name + "L")
		printDecisionTree(rec+1, n.M, name + "M")
		printDecisionTree(rec+1, n.R, name + "R")
	elif (n.type == "leaf"):
		print(spaces + "leaf " + name + " :")
		#print(n.inlier)
		for value in n.inlier:
			print(spaces + "\t" + str(value))
	else:
		print(spaces + "ERROR !")



node = buildDecisionTree(data, True, [0,1], data)
printDecisionTree(0, node)


