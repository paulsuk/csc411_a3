import a3
import matplotlib.pyplot as plt

def part4(classifier, num_iterations=250, alpha=0.00003):
	print("perfomring number 4")
	train_perf, test_perf, _ = classifier.LogisticRegression(num_iterations=num_iterations,
		alpha=alpha, part4=True)

	plt.figure()
	plt.title("Part 4: Using LogisticRegression for NLP")
	plt.ylabel("Performance")
	plt.xlabel("Iterations")
	plt.plot(train_perf, label="Training Accuracy")
	plt.plot(test_perf, label="Testing Accuracy")
	plt.legend(loc=4)
	plt.savefig("Part4_logreg_perf.png")

if __name__ == '__main__':
	classes = {}
	dir_base = "txt_sentoken/"
	classes["positive"] = dir_base + "pos/"
	classes["negative"] = dir_base + "neg/"
	classifier = a3.Classifier(classes)
	print("initialized classes")

	# Run the actual parts
	part4(classifier)
