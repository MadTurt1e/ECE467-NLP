import math
import string
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from tqdm import tqdm


# I did Naive Bayes because it is straightforward to implement - "bag o words"
class NaiveBayesClassifier:
    def __init__(self, smoothing_constant=0.015):
        self.vocab = set()
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.category_word_counts = defaultdict(int)
        self.category_doc_counts = defaultdict(int)
        self.total_docs = 0
        self.stop_words = set(stopwords.words("english"))
        self.smoothing_constant = smoothing_constant

    def lemmatizer(self, word):
        # Use a lemmatizer because it may be more accurate
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(word.lower())
        # If the word is a "useless" word we ignore it
        return [lemmatizer.lemmatize(token) for token in tokens
                if token not in self.stop_words
                and token not in string.punctuation]

    def train(self, documents, categories):
        # We record the amount of actual documents there are
        self.total_docs = len(documents)

        # Iterate through each document and category (zip is a way to do this)
        # I learned about tqdm in DL and now it is necessary everywhere
        for doc, category in tqdm(zip(documents, categories),
                                  total=len(documents), desc="  Training"):
            # Increase the count of documents of category
            self.category_doc_counts[category] += 1
            # sets avoid "double counting" in one document
            words = self.lemmatizer(doc)

            # Count the words
            for word in words:
                self.vocab.add(word)
                self.word_counts[category][word] += 1
                self.category_word_counts[category] += 1

    # Naive Bayes slideshow implementation instead
    def predict(self, document):
        words = self.lemmatizer(document)
        scores = {}

        # Calculate a probability score for our document
        for category in self.category_doc_counts:
            # Initialize the score as log probability of each category
            score = math.log(self.category_doc_counts[category]
                             / self.total_docs)

            # Go through every word
            for word in words:
                # Find prob of word appearing and add it to bayes score
                word_count = self.word_counts[category][word]
                total_words = self.category_word_counts[category]

                # Improved probability estimation with Laplace smoothing
                word_prob = (word_count + self.smoothing_constant) / (
                    total_words + (len(self.vocab) * self.smoothing_constant)
                )
                score += math.log(word_prob)
            scores[category] = score

        # Return the category with the highest score
        return max(scores, key=scores.get)


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def main():
    # First we get input from the user
    train_file = input("Input training file name: ")
    test_file = input("Input testing file name: ")

    # # Training and testing file that we use for ease
    # train_file = "corpus1_train.labels"
    # test_file = "corpus1_test.list"

    # Read training data
    train_documents = []
    train_categories = []

    # Open training file
    with open(train_file, "r") as f:
        # For every line, split it into filename and category
        for line in f:
            file_path, category = line.strip().split()
            train_documents.append(read_file(file_path))
            train_categories.append(category)

    # Make a classifier
    classifier = NaiveBayesClassifier()
    # Launch our dataset into our classifier
    classifier.train(train_documents, train_categories)

    # Read test data
    test_documents = []
    test_file_paths = []
    with open(test_file, "r") as f:
        for line in f:
            # # Use this if you're testing on something labeled
            # file_path, category = line.strip().split()
            file_path = line.strip()
            test_documents.append(read_file(file_path))
            test_file_paths.append(file_path)

    # Make predictions
    predictions = []
    for doc in tqdm(test_documents, desc="Predicting"):
        predictions.append(classifier.predict(doc))

    # Write output
    output_file = input("Input output file name: ")
    with open(output_file, "w") as f:
        for file_path, prediction in zip(test_file_paths, predictions):
            f.write(f"{file_path} {prediction}\n")

    print(f"Output to {output_file}")


if __name__ == "__main__":
    main()
