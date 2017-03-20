import a3
import matplotlib.pyplot as plt
import numpy as np

def countWords(word, data):
	count = 0
	for wrd in data:
		if wrd == word:
			count += 1
	return count

def part1(classifier):
	'''
	Prints the frequency of a few select words in the training set
	'''
	pos_words = classifier.classes["positive"]._train_set.words()
	neg_words = classifier.classes["negative"]._train_set.words()

	words = ["good", "amazing", "inspirational", "awful", "bad", "boring"]

	for word in words:
		count_pos = countWords(word, pos_words)
		count_neg = countWords(word, neg_words)

		print("{} occurs {} times in positive reviews and {} times in negative reviews".format(word, count_pos, count_neg))

def part2(classifier):
	'''
	Compares the performance of the Naive Bayes Classifier with respect to changing m
	'''
	print("running part 2")
	m = 0.00000

	ms = []
	test_perf = []
	train_perf = []

	for i in range(25):
		ms.append(m)
		total_train, train_corr, total_test, test_corr = classifier.NaiveBayes(m)
		train_perf.append(train_corr/total_train)
		test_perf.append(test_corr/total_test)

		m += 0.00002

		print("Naive Bayes Classifier Performance for m = {}: ".format(m))
		print("Training: {} of {}, {}%, Testing: {} of {}, {}%".format(train_corr, total_train, 
				(100*train_corr/total_train), test_corr, total_test, (100*test_corr/total_test)))
	
	plt.figure()
	plt.plot(ms, train_perf, 'r', label="Training performance")
	plt.plot(ms, test_perf, 'b', label="Testing performance")
	plt.axis([0, ms[-1], 0, 1])
	plt.xlabel("m")
	plt.ylabel("Performance")
	plt.title("Part 2: Naive Bayes Performance on m")
	plt.legend()
	plt.savefig("part2_m_graph.png")

def part3(classifier, n=10):
	'''
	prints the words that have the highest correlation with each class
	'''
	print("running part 3")
	m = 0.00005
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


	likely_pos = [pos_words[i] for i in ix_pos]
	likely_neg = [neg_words[i] for i in ix_neg]

	print("The {} most likely words for positive reviews are: {}".format(n, likely_pos))
	print("The {} most likely words for negative reviews are: {}".format(n, likely_neg))


if __name__ == '__main__':
	classes = {}
	dir_base = "txt_sentoken/"
	classes["positive"] = dir_base + "pos/"
	classes["negative"] = dir_base + "neg/"
	classifier = a3.Classifier(classes)
	print("initialized classes")

	# Run the actual parts
	part1(classifier)
	part2(classifier)
	part3(classifier)
