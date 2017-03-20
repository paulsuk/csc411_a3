import a3
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import tensorflow as tf
import pdb

	def part7(self, num_iterations=250, alpha=0.001):
		train_perf, test_perf = self.LogisticRegression(num_iterations=num_iterations,
			alpha=alpha, part4=False)

		plt.figure()
		plt.title("Part 7: Using LogisticRegression for word2vec")
		plt.ylabel("Performance")
		plt.xlabel("Iterations")
		plt.plot(train_perf, label="Training Accuracy")
		plt.plot(test_perf, label="Testing Accuracy")
		plt.legend(loc=4)
		plt.savefig("Part7_logreg_perf.png")

def part8(classifier):
		pos_review_class = classifier.classes["positive"]
		neg_review_class = classifier.classes["negative"]
		all_reviews = pos_review_class._train_set + neg_review_class._train_set

		words = ['story', 'good', 'oscar', 'plotholes']

		closest_story = all_reviews.isInContext_embeddings_count(words[0])
		closest_good = all_reviews.isInContext_embeddings_count(words[1])
		closest_oscar = all_reviews.isInContext_embeddings_count(words[2])
		closest_plotholes = all_reviews.isInContext_embeddings_count(words[3])

		print ("The 10 closest words with highest frequency to {} are {}".format(words[0], closest_story))
		print ("The 10 closest words with highest frequency to {} are {}".format(words[1], closest_good))
		print ("The 10 closest words with highest frequency to {} are {}".format(words[2], closest_oscar))
		print ("The 10 closest words with highest frequency to {} are {}".format(words[3], closest_plotholes))

if __name__ == '__main__':

	classes = {}
	dir_base = "txt_sentoken/"

	classes["positive"] = dir_base + "pos/"
	classes["negative"] = dir_base + "neg/"

	classifier = a3.Classifier(classes)
	print("initialized classes")

	#classifier.part7()
	classifier.part8()