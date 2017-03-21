import a3
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import tensorflow as tf
import pdb

INF = 10000

emb = np.load("embeddings.npz")["emb"]
word2ind = np.load("embeddings.npz")["word2ind"].flatten()[0]

def part7(classifier, num_iterations=250, alpha=0.001):
	train_perf, test_perf = classifier.LogisticRegression(num_iterations=num_iterations,
		alpha=alpha, part4=False)

	plt.figure()
	plt.title("Part 7: Using LogisticRegression for word2vec")
	plt.ylabel("Performance")
	plt.xlabel("Iterations")
	plt.plot(train_perf, label="Training Accuracy")
	plt.plot(test_perf, label="Testing Accuracy")
	plt.legend(loc=4)
	plt.savefig("Part7_logreg_perf.png")

def euclid_dist(word, des):
	'''
	Calculate Euclidean distance
	'''
	return np.linalg.norm(emb[word] - emb[des])

def cos_dist(word, des):
	'''
	Calculate Cosine Distance
	'''
	word = emb[word]
	des = emb[des]
	return 1.0 - np.dot(word, des)/(np.linalg.norm(word)*np.linalg.norm(des))

def top10(des_index, isEuclid):
	'''
	Given the index of a word, return a list
	of top10 closest words
	'''
	num_indicies = emb.shape[0]	
	
	top10_dist = np.empty(10) #Stores distances
	top10_dist[:] = INF

	top10_words = [0]*10 #Stores words
		
	for word in range(num_indicies):
		#Ignore itself
		if word == des_index:
			continue
		
		if isEuclid:
			dist = euclid_dist(word, des_index)
		else:
			dist = cos_dist(word, des_index)
		
		if dist < np.max(top10_dist):
			i = np.argmax(top10_dist)
			top10_dist[i] = dist
			top10_words[i] = word2ind[word]

	return top10_words

def part8():
	index = [67, 60, 1497, 77] #Indexes for story, good, award, character

	for i in index:
		ls = top10(i, True)
		print ("The closest 10 words (euclid dist) to '{}' are".format(word2ind[i]))
		print (ls)

		ls = top10(i, False)
		print ("The closest 10 words (cosine dist) to '{}' are".format(word2ind[i]))
		print (ls)

if __name__ == '__main__':
	'''
	classes = {}
	dir_base = "txt_sentoken/"

	classes["positive"] = dir_base + "pos/"
	classes["negative"] = dir_base + "neg/"

	classifier = a3.Classifier(classes)
	print("initialized classes")

	part7(classifier)	
	#part8(classifier)'''
	part8()