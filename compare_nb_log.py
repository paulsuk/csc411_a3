import a3
import matplotlib.pyplot as plt
import numpy as np
import tabulate
import pdb

def part6(classifier):
	print("performing part 6")
	# Get thetas for logistic:
	n = 100
	num_iterations = 250
	alpha = 0.00003

	_, _, thetas = classifier.LogisticRegression(num_iterations=num_iterations,
		alpha=alpha, part4=True)

	ix = np.argsort(thetas)[ :-n -1: -1]

	theta_log = np.array([(classifier._vocabulary[i], thetas[i]) for i in ix])

	# NB part
	m = 0.00005
	pos_class = classifier.classes["positive"]
	neg_class = classifier.classes["negative"]

	words = classifier._vocabulary

	pos_prob = np.array(pos_class.probabilities_of_words(words, m))
	neg_prob = np.array(neg_class.probabilities_of_words(words, m))

	ratio = pos_prob/neg_prob

	ix_nb = np.argsort(ratio)[ :-n -1: -1]

	theta_nb = np.array([(words[i], ratio[i]) for i in ix_nb])

	pdb.set_trace()
	top_theta = np.append(theta_nb, theta_log, axis=1)
	print(tabulate.tabulate(top_theta, headers=["nb_word", "nb_theta", "log_word", "log_theta"], 
		numalign="center", tablefmt="psql"))

if __name__ == '__main__':
	classes = {}
	dir_base = "txt_sentoken/"
	classes["positive"] = dir_base + "pos/"
	classes["negative"] = dir_base + "neg/"
	classifier = a3.Classifier(classes)
	print("initialized classes")

	part6(classifier)