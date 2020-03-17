from prepare_data import prepare_data, split_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

dataset = prepare_data()

vectorizer = CountVectorizer()


def mean_accuracy(dataset, spam_proportion=0):
    accuracies = []

    for part_number in range(len(dataset)):
        train_features, train_labels, test_features, test_labels = split_data(part_number, dataset)

        train_word_counts = vectorizer.fit_transform(train_features)

        if spam_proportion == 0:
            model = MultinomialNB()
        else:
            model = MultinomialNB(fit_prior=False, class_prior=[spam_proportion, 1])

        model.fit(train_word_counts, train_labels)

        test_word_counts = vectorizer.transform(test_features)

        accuracies.append(accuracy_score(test_labels, model.predict(test_word_counts)))

    print(sum(accuracies)/len(accuracies))

    return sum(accuracies)/len(accuracies)


def roc(dataset):
    squashed_dataset = []

    for part in dataset:
        for message in part:
            squashed_dataset.append(message)

    squashed_dataset = np.array(squashed_dataset)
    dataset = squashed_dataset

    train_features = dataset[:, 0]
    train_labels = dataset[:, 1]

    test_features = train_features
    test_labels = train_labels

    train_word_counts = vectorizer.fit_transform(train_features)

    model = MultinomialNB()
    model.fit(train_word_counts, train_labels)

    test_word_counts = vectorizer.transform(test_features)

    false_positive_rate, true_positive_rate, threshold = roc_curve(test_labels, model.predict_proba(test_word_counts)[:, 0], pos_label='0')

    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], color='lightgrey', linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def zero_false_positives(dataset):
    squashed_dataset = []

    for part in dataset:
        for message in part:
            squashed_dataset.append(message)

    squashed_dataset = np.array(squashed_dataset)
    dataset = squashed_dataset

    train_features = dataset[:, 0]
    train_labels = dataset[:, 1]

    test_features = train_features
    test_labels = train_labels

    train_word_counts = vectorizer.fit_transform(train_features)

    model = MultinomialNB(fit_prior=False, class_prior=[
        0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000001, 1])
    model.fit(train_word_counts, train_labels)

    test_word_counts = vectorizer.transform(test_features)

    false_positive_rate, true_positive_rate, threshold = roc_curve(test_labels,
                                                                   model.predict_proba(test_word_counts)[:, 0],
                                                                   pos_label='0')

    false_positive_count = sum([1 for actual, predicted in zip(test_labels, model.predict(test_word_counts)) if
                                (actual != predicted) and (actual == '1')])

    print(false_positive_count)


def mean_accuracy_progression(dataset, min_spam_proportion):
    spam_proportion = 1
    accuracies = []
    spam_proportions = []

    while spam_proportion >= min_spam_proportion:
        accuracies.append(mean_accuracy(dataset, spam_proportion))

        spam_proportions.append(spam_proportion)

        spam_proportion = spam_proportion / 10000000000

    plt.title('Accuracy')
    plt.plot(accuracies, spam_proportions)
    plt.ylabel('Accuracy')
    plt.xlabel('Spam/Legit')
    plt.show()

min_spam_proportion = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000001

#mean_accuracy_progression(dataset, min_spam_proportion)

roc(dataset)