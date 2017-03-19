import math
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import tensorflow as tf
import pdb

class Review(object):
	'''
	based off of http://www.python-course.eu/text_classification_python.php
	Representation of a review, storing the total number of words, as well as the frequency of each word
	'''

	def __init__(self):
		# word_count is the frequency of words in the review
		self._num_words = 0
		self._word_count = {}
		self._all_words = []
		self._vocab_size = 0

	def __add__(self, other):
		'''
		Overriding the '+' operator when joining two Reviews
		'''
		res = Review()
		for word in self._word_count:
			res._word_count[word] = self._word_count[word]
			if word in other._word_count:
				res._word_count[word] += other._word_count[word]
		for word in other._word_count:
			if word not in res._word_count:
				res._word_count[word] = other._word_count[word]
		res._num_words = self._num_words + other._num_words
		res._all_words = self._all_words + other._all_words
		res._vocab_size = self._vocab_size + other._vocab_size

		return res

	def read_file(self, filename):
		'''
		Read file and update the words of the review
		'''
		translator = str.maketrans("", "", string.punctuation)
		file = open(filename).read().lower().replace("\n", "").translate(translator)
		words = file.split()
		self._all_words = words
		for word in words:
			self._add_word(word)

		for word in self._word_count:
			self._word_count[word] *=(1/len(words))
		self._vocab_size = len(self._word_count)/len(words)
		
	def _add_word(self, word):
		'''Add word to review'''
		self._num_words += 1
		if word in self._word_count:
			self._word_count[word] += 1
		else:
			self._word_count[word] = 1

	def total_num_words(self):
		return self._num_words

	def vocabulary_size(self):
		''' Return the number of unique words in Review'''
		return self._vocab_size

	def words(self):
		''' Returns the list of words contained in object including duplicates'''
		return self._all_words

	def unique_words(self):
		'''
		returns a list of all of the unique words
		'''
		return list(self._word_count.keys())

	def words_and_count(self):
		''' Gets the word frequency dictionary and returns it'''
		return self._word_count

	def wordCount(self, word):
		''' Returns the frequency of a word in the review '''
		if word in self._word_count:
			return self._word_count[word]
		else:
			return 0
		

class ReviewClass(object):
	'''
	A class of reviews
	'''
	def __init__(self, className):
		self.class_name = className

		self._reviews = []
		self._reviews_test = []
		self._reviews_val = []
		self._train_set = Review()

	def read_from_dir(self, directory):
		'''
		Reads all of the files in a directory, adds them to self, 
		with each file being a Review object, using read_file

		also creates a review that grouped all of the reviews into a single Review class, stores it 
		in self._all_reviews
		'''
		np.random.seed(100)
		filenames = np.array(os.listdir(directory))
		np.random.shuffle(filenames)
		train_set = Review()

		for i in range(filenames.size):
			file = directory + filenames[i]
			review = Review()
			review.read_file(file)
			if i < 600:
				self._reviews.append(review)
				train_set += review
			elif i < 800:
				self._reviews_test.append(review)
			else:
				self._reviews_val.append(review)

		self._train_set = train_set

	def num_of_reviews(self):
		return len(self._reviews)

	def num_words_in_class(self):
		'''
		returns the total number of words in class
		'''
		reviews = self._reviews
		count = 0

		for review in reviews:
			count += review.total_num_words()

		return count

	def words_in_class(self):
		return self._train_set.unique_words()

	def class_vocab_size(self):
		'''
		Returns the vocabulary size of the entire class
		'''
		return self._train_set.vocabulary_size()

	def probability_of_word(self, word, m):
		'''
		Returns the probability of word given it being a review of this class
		'''
		k = self.class_vocab_size()
		tot = self.num_of_reviews()

		if not self._train_set:
			return float((m*k)/(tot + k))
		else:
			count = self._train_set.wordCount(word)
			return float((count + m*k)/(tot+k))

	def probabilities_of_words(self, words, m):
		probabilities = []

		for word in words:
			probabilities.append(self.probability_of_word(word, m))
		return np.array(probabilities)


class Classifier(object):
	'''
	The classifier object, holds ReviewClass objects and will do a classificaiton
	'''
	def __init__(self, classes):
		'''
		classes is a dictioanry where the keys are classes, and their value is the directory of the 
		location of the files of that class
		'''
		self.classes = {}
		for class_name in classes:
			self._add_class(class_name, classes[class_name])

		self._vocabulary = [] 

	def _add_class(self, class_name, directory):
		'''
		Creates a ReviewClass and adds it to self.classes
		'''
		if class_name in self.classes:
			print("classes must have unique names")
			return

		reviewClass = ReviewClass(class_name)
		reviewClass.read_from_dir(directory)

		self.classes[class_name] = reviewClass

	def _argMax_bayes(self, words, m):
		'''
		- returns the classname that maximizes the naive bayesian probability
		- assumes the probability of being in any class is the same
		'''
		argmax_p = -math.inf
		argmax_class = None

		num_classes = len(self.classes)

		# Get the bayesian probabiliy, find the max class
		for class_name in self.classes:
			reviewClass = self.classes[class_name]

			p = 0
			for word in words:
				p += np.log(reviewClass.probability_of_word(word, m))
			# assuming each class is equally likely to occur
			p = p/num_classes

			if p > argmax_p:
				argmax_p = p
				argmax_class = class_name

		return argmax_class 

	def _build_x_from_review(self, review):
		x = []
		for word in self._vocabulary:
			count = review.wordCount(word)
			if count > 0:
				x.append(1)
			else:
				x.append(0)
		return np.array(x)

	def get_data(self):
		'''
		Uses _build_x_from_review to build x, y, vectors
		x will be n x k
		y will be n x 1
		'''

		x = np.array([])
		x_t = np.array([])
		y = np.array([])
		y_t = np.array([])

		for class_name in self.classes:
			label = 0
			label += (class_name == "positive")

			reviewClass = self.classes[class_name]
			for review in reviewClass._reviews:
				x_temp = self._build_x_from_review(review)
				if x.size == 0:
					x = x_temp
					y = np.array([label])
				else:
					x = np.vstack((x, x_temp))
					y = np.append(y, [label])
			for review in reviewClass._reviews_test:
				x_temp = self._build_x_from_review(review)
				if x_t.size == 0:
					x_t = x_temp
					y_t = np.array([label])
				else:
					x_t = np.vstack((x_t, x_temp))
					y_t = np.append(y_t, [label])
		return x, y, x_t, y_t

	def NaiveBayes(self, m):
		total_train = 0
		total_test = 0

		train_corr = 0
		test_corr = 0

		for class_name in self.classes:
			reviewClass = self.classes[class_name]

			reviews_train = reviewClass._reviews
			reviews_test = reviewClass._reviews_test
			total_train += len(reviews_train)
			total_test += len(reviews_test)

			for review in reviews_train:
				words = review.words()
				guess = self._argMax_bayes(words, m)
				if guess == class_name:
					train_corr += 1
			for review in reviews_test:
				words = review.words()
				guess = self._argMax_bayes(words, m)
				if guess == class_name:
					test_corr += 1
		return total_train, train_corr, total_test, test_corr

	def LogisticRegression(self):
		'''
		Will use tensorflow to do a logistic regression
		'''
		pos_review_class = self.classes["positive"]
		neg_review_class = self.classes["negative"]
		
		all_reviews = pos_review_class._train_set + neg_review_class._train_set

		self._vocabulary = all_reviews.unique_words()
		print("getting data")
		x, y, x_t, y_t = self.get_data()

		# TODO: Do tensorflow network here to do logistic regression
		# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py


	def part2(self):
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
			total_train, train_corr, total_test, test_corr = self.NaiveBayes(m)
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

	def part3(self, n=10):
		'''
		prints the words that have the highest correlation with each class
		'''
		print("running part 3")
		m = 0.0005
		pos_class = self.classes["positive"]
		neg_class = self.classes["negative"]

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

	classifier = Classifier(classes)
	print("initialized classes")
	#classifier.part2()
	#classifier.part3(n=10)
	classifier.LogisticRegression()
