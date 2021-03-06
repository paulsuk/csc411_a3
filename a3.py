import math
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import tensorflow as tf
import pdb

class Review(object):
	'''
	-Representation of a review, storing the total number of words, 
	as well as the frequency of each word, vocabulary size

	- This class supports being able to add to other reviews
	'''

	def __init__(self):
		'''
		_num_words is the total number of words
		_word_count is a dict with key being words, and count is the normalized occurance of word
		_all_words is the review word by word
		_vocab_size is the number of unique words 
		Note that the num words will be normalized to 1 for a single review,
		and will represent the total number of reviews combined to make the object
		'''
		self._num_words = 0
		self._word_count = {}
		self._all_words = []
		self._vocab_size = 0

	def __add__(self, other):
		'''
		Overriding the '+' operator when joining two Reviews
		Combines the word_count dictionary
		Sums the num words, all_words
		Updates vocab size
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
		Reads textfile representing a review, preprocesses the words
			- removes punctuation
			- seperates word by word
			- makes all lowercase
		Updates corresponding values in class
		Normalizes vocab sizes, counts, by the total length of review
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
		'''
		Add word to review, is called by read_file
		Note that while the count is incremented by 1, it is normalized 
		at the end of read_file(). Don't call this method directly
		'''
		self._num_words += 1
		if word in self._word_count:
			self._word_count[word] += 1
		else:
			self._word_count[word] = 1

	def total_num_words(self):
		'''
		Number of words (non-unique)
		'''
		return self._num_words

	def vocabulary_size(self):
		'''
		Return the number of unique words in Review
		'''
		return self._vocab_size

	def words(self):
		'''
		Returns the list of words contained in object including duplicates
		'''
		return self._all_words

	def unique_words(self):
		'''
		returns a list of all of the unique words
		'''
		return list(self._word_count.keys())

	def wordCount(self, word):
		'''
		Returns the frequency of a word in the review
		'''
		if word in self._word_count:
			return self._word_count[word]
		else:
			return 0

	def make_context(self):
		words = self.words()
		context = []
		for i in range((self.total_num_words())-1):
			context.append((words[i], words[i+1]))
		return context

	def make_context_embeddings(self):
		word_indices = np.load("embeddings.npz")["word2ind"].flatten()[0]
		context = []
		for i in range(len(word_indices)-1):
			context.append((word_indices[i], word_indices[i+1]))
			context.append((word_indices[i+1], word_indices[i]))
		return context

	def make_context_embeddings_sample(self):
		word_indices = np.load("embeddings.npz")["word2ind"].flatten()[0]
		context = []
		for i in range(2):
			context.append((word_indices[i], word_indices[i+1]))
			context.append((word_indices[i+1], word_indices[i]))

		print ("Made context! {}".format(context))
		return context
	
	def isInContext(self, word):
		''' Returns true if word is in context ''' 
		context = self.make_context()
		for x in context:
			if x[1] == word:
				return True
		return False

	def isInContext_embeddings(self, word):
		''' Returns true if word is in context'''
		context = self.make_context_embeddings_sample()
		for x in context:
			if x[1] == word:
				print ("is in context!")
				return True
		return False

	def isInContext_embeddings_count(self, word):
		''' Returns list of 10 words closest to the embedding of 'word' '''
		context = self.make_context()#_embeddings()
		print ("Made context")
		close_word = {}
		for x in context:
			if x[1] == word: 
				if x[0] not in close_word.keys():
					close_word[x[0]] = 1
				else:
					close_word[x[0]] += 1
		sorted_close = sorted(close_word.items(), key=lambda x: x[1], reverse=True)
		print (sorted_close)
		closest = [x[0] for x in sorted_close[:10]]
		return closest


class ReviewClass(object):
	'''
	A collection of Review objects that are in the same class
	- is initialized from a directory, contains the training, testing, validation set
	- also has _train_set() which is a Review() object which is all the reviews
		in the training set grouped into 1 Review object for some simplicity
	- Used in the Bayesian section to get the probability of a word given being in this class
	'''
	def __init__(self, className):
		'''
		Initialize class name, train/test/val sets, and the _train_set object
		'''
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
		in self._test_set
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
		'''
		Returns the size of the training set 
		'''
		return len(self._reviews)

	def words_in_class(self):
		'''
		Returns the unique words in the training set
		'''
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
		'''
		Returns an array of probabilities using the probability_of_word method
		'''
		probabilities = []

		for word in words:
			probabilities.append(self.probability_of_word(word, m))
		return np.array(probabilities)


class Classifier(object):
	'''
	The classifier object, holds ReviewClass objects and will do a classificaiton
	Holds a dictionary of class names mapped to their corresponding ReviewClass 
	- Also has the vocabulary of the entire training set
	'''
	def __init__(self, classes):
		'''
		classes is a dictioanry where the keys are classes, and their value is the directory of the 
		location of the files of that class
		vocabulary is the combined vocabulary of the entire dataset
		'''
		self.classes = {}
		for class_name in classes:
			self._add_class(class_name, classes[class_name])

		self._vocabulary = [] 

	def _add_class(self, class_name, directory):
		'''
		Creates a ReviewClass and adds it to self.classes
		Reads all of the files in the given directory and adds them to ReviewClass
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
		- assumes the probability of being in any class is the same (1/2)
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
		'''
		Used for part 4, builds the feature vector for a review
		- each vector is k dimensional, where k is the size of total vocabulary
		- if the kth word is in review ever, the value is 1, otherwise it is 0
		'''
		x = []
		for word in self._vocabulary:
			count = review.wordCount(word)
			if count > 0:
				x.append(1)
			else:
				x.append(0)
		return np.array(x)

	def _build_x_from_embeddings(self, review):
		x = []
		for word in self._vocabulary:
			if review.isInContext_embeddings(word):
				x.append(1)
			else:
				x.append(0)
		print ("DONE BUILDING X")
		return np.array(x)

	def get_data_embeddings(self):
		'''

		Uses _build_x_from_review to build x, y, vectors
		- takes the entire training set from all classes to build the input vectors

		x will be n x k
		y will be n x 1
		'''

		x = np.array([])
		x_t = np.array([])
		y = np.array([])
		y_t = np.array([])

		for class_name in self.classes:
			if class_name == "positive":
				label = np.array([1, 0])
			else: 
				label = np.array([0, 1])

			reviewClass = self.classes[class_name]
			for review in reviewClass._reviews[:1]:
				x_temp = self._build_x_from_embeddings(review)
				if x.size == 0:
					x = x_temp
					y = np.array([label])
				else:
					x = np.vstack((x, x_temp))
					y = np.vstack((y, label))
			for review in reviewClass._reviews_test[:1]:
				x_temp = self._build_x_from_embeddings(review)
				if x_t.size == 0:
					x_t = x_temp
					y_t = np.array([label])
				else:
					x_t = np.vstack((x_t, x_temp))
					y_t = np.vstack((y_t, label))
		return x, y, x_t, y_t

	def get_data(self):
		'''
		Uses _build_x_from_review to build x, y, vectors
		- takes the entire training set from all classes to build the input vectors
		x will be n x k
		y will be n x 1
		'''

		x = np.array([])
		x_t = np.array([])
		y = np.array([])
		y_t = np.array([])

		for class_name in self.classes:
			if class_name == "positive":
				label = np.array([1, 0])
			else: 
				label = np.array([0, 1])

			reviewClass = self.classes[class_name]
			for review in reviewClass._reviews:
				x_temp = self._build_x_from_review(review)
				if x.size == 0:
					x = x_temp
					y = np.array([label])
				else:
					x = np.vstack((x, x_temp))
					y = np.vstack((y, label))
			for review in reviewClass._reviews_test:
				x_temp = self._build_x_from_review(review)
				if x_t.size == 0:
					x_t = x_temp
					y_t = np.array([label])
				else:
					x_t = np.vstack((x_t, x_temp))
					y_t = np.vstack((y_t, label))
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

	def LogisticRegression(self, num_iterations=250, alpha=0.0001, part4=True):
		'''
		Will use tensorflow to do a logistic regression
		'''
		pos_review_class = self.classes["positive"]
		neg_review_class = self.classes["negative"]
		
		all_reviews = pos_review_class._train_set + neg_review_class._train_set

		self._vocabulary = all_reviews.unique_words()
		print("getting data")

		if part4:
			x, y, x_t, y_t = self.get_data()
		else:
			x, y, x_t, y_t = self.get_data_embeddings()

		print("starting tensorflow")
		# parameters
		k = x.shape[1]
		m = y.shape[1]
		beta = 0.1/(x.shape[0])

		# placeholders
		_x = tf.placeholder(tf.float32, [None, k]) # k dimensional vector
		_y = tf.placeholder(tf.float32, [None, m]) # labels

		# model weights
		w = tf.Variable(tf.random_normal([k, m], stddev=0.01))
		b = tf.Variable(tf.random_normal([m], stddev=0.01))

		# softmax
		pred = tf.nn.softmax(tf.matmul(_x, w) + b)

		# cost
		cost = tf.reduce_mean(-tf.reduce_sum(_y*tf.log(pred)))
		regularization = tf.nn.l2_loss(w)
		cost = tf.reduce_mean(cost + beta * regularization)

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

		init = tf.initialize_all_variables()
		sess = tf.Session()
		sess.run(init)

		#accuracy
		correct_pred = tf.equal(tf.argmax(_y, 1), tf.argmax(pred, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		train_acc = []
		test_acc = []

		for i in range(num_iterations):
			sess.run(optimizer, feed_dict={_x: x, _y: y})
			train_acc.append(sess.run(accuracy, feed_dict={_x: x, _y: y}))
			test_acc.append(sess.run(accuracy, feed_dict={_x: x_t, _y: y_t}))

			if i % 10 == 0:
				print("iteration " + str(i))
				print("Train: {}".format(sess.run(accuracy, feed_dict={_x: x, _y: y})))
				print("Test: {}".format(sess.run(accuracy, feed_dict={_x: x_t, _y: y_t})))

		return train_acc, test_acc, w.eval(sess)[:,0]

if __name__ == '__main__':

	classes = {}
	dir_base = "txt_sentoken/"

	classes["positive"] = dir_base + "pos/"
	classes["negative"] = dir_base + "neg/"

	classifier = a3.Classifier(classes)
	print("initialized classes")
	#classifier.part2()
	#classifier.part3(n=10)
