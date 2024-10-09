import math
import string
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

from tqdm import tqdm


# I did Naive Bayes because it is straightforward to implement - "bag o words"
class NaiveBayesClassifier:
    def __init__(self):
        # First, we set up the vocabulary as a "python array" with nothing
        # (of course we don't do arrays in python, but let's say we do)
        self.vocab = set()

        # We split the word counter into a double "array", where all values
        # are initialized as 0 (by defaultdict)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        # We set up an "array" of all the categories, so we can figure out
        # the documents spotted so far
        self.category_counts = defaultdict(int)
        # We also want to keep track of the total amount of documents seen
        self.total_docs = 0
        # Also there's a function that eliminates all "useless" words
        self.stop_words = set(stopwords.words("english"))

    def lemmatizer(self, word):
        # Use a lemmatizer because it may be more accurate
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(word.lower())
        # If the word is a "useless" word we ignore it
        return [lemmatizer.lemmatize(token) for token in tokens
                if token not in self.stop_words
                and token not in string.punctuation]

    # Synynoms can also be relevant
    def get_synynoms(self, word):
        synonyms = []
        # Hunt through every synynom we find
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # I notice synynoms have underscores for some reason
                synonym = lemma.name().lower().replace("_", " ")
                # If we have literally saw the synynom already ignore it
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        # We now have a list of synynoms
        return synonyms

    def train(self, documents, categories):
        # We record the amount of actual documents there are
        self.total_docs = len(documents)

        # Iterate through each document and category (zip is a way to do this)
        # I learned about tqdm in DL and now it is necessary everywhere
        for doc, category in tqdm(zip(documents, categories),
                                  total=len(documents), desc="  Training"):
            # Mention to our NaiveBayes: hey, there's a document in here
            self.category_counts[category] += 1
            # We just lemmatize straight now
            words = self.lemmatizer(doc)

            # Iterate through every token
            for word in words:
                # Note down that this is a word
                self.vocab.add(word)

                # Add this to (specifically) our category word count
                self.word_counts[category][word] += 1

                # We can also add all the synynoms
                synynoms = self.get_synynoms(word)
                for synynom in synynoms:
                    self.vocab.add(synynom)
                    # Synynoms are slightly less meaningful, but I find they
                    # should be meaningful depending on category count
                    synynom_meaningfulness = 0.7
                    if len(self.word_counts) < 4:
                        synynom_meaningfulness = 0.2

                    self.word_counts[
                        category][synynom] += synynom_meaningfulness

    # https://en.wikipedia.org/wiki/Naive_Bayes_classifier
    # Naive Bayes is basically an implementation of Wikipedia
    def predict(self, document):
        words = self.lemmatizer(document)
        scores = {}

        # Calculate a probability score for our document
        for category in self.category_counts:
            # Initialize the score as log probability of each category
            score = math.log(self.category_counts[category] / self.total_docs)
            # Go through every word
            for word in words:
                # If the word exists in our vocabulary we find the prob of it
                # appearing and add it to our bayes score
                if word in self.vocab:
                    word_count = self.word_counts[category][word]
                    # Laplace smoothing from wikipedia
                    score += math.log(
                        (word_count + 1)
                        / (sum(self.word_counts[category].values())
                           + len(self.vocab))
                    )
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
