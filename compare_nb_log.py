import a3
import matplotlib.pyplot as plt
import numpy as np
import pdb

def part6(classifier):
	print("performing part 6")
	# NB part
	n = 100
	m = 0.000005
	pos_class = classifier.classes["positive"]
	neg_class = classifier.classes["negative"]

	pos_words = pos_class.words_in_class()
	neg_words = neg_class.words_in_class()

	pos_in_pos = np.log(pos_class.probabilities_of_words(pos_words, m))
	pos_in_neg = np.log(neg_class.probabilities_of_words(pos_words, m))

	neg_in_pos = np.log(pos_class.probabilities_of_words(neg_words, m))
	neg_in_neg = np.log(neg_class.probabilities_of_words(neg_words, m))

	pos_diff = pos_in_pos - pos_in_neg
	neg_diff = neg_in_neg - neg_in_pos

	ix_pos = np.argsort(pos_diff)[ :-n -1: -1]
	ix_neg = np.argsort(neg_diff)[ :-n -1: -1]

	theta_nb = [pos_diff[i] for i in ix_pos]

	# Get thetas for logistic:
	num_iterations = 250
	alpha = 0.00003

	_, _, thetas = classifier.LogisticRegression(num_iterations=num_iterations,
		alpha=alpha, part4=True)

	ix = np.argsort(thetas)[ :-n -1: -1]

	theta_log = [thetas[i] for i in ix]
	pdb.set_trace()
	print("yo")

if __name__ == '__main__':
	classes = {}
	dir_base = "txt_sentoken/"
	classes["positive"] = dir_base + "pos/"
	classes["negative"] = dir_base + "neg/"
	classifier = a3.Classifier(classes)
	print("initialized classes")

	part6(classifier)