from prepare_data import prepare_data, split_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def mean_accuracy_plotter(dataset):
    # полагаем изначально долю спама в сообщениях 50% и она падает до MIN_SPAM_PROPORTION (априорная вероятность встретить спам), в зависимости от этого получаем accuracy
    spam_proportion = 0.5
    accuracies = []
    spam_proportions = []
    while spam_proportion >= MIN_SPAM_PROPORTION:
        # собираем средние значения точности классификации для данной априорной вероятности спама
        accuracies.append(mean_accuracy(dataset, spam_proportion))
        spam_proportions.append(spam_proportion)
        spam_proportion -= 0.05
    plt.plot(spam_proportions, accuracies)
    print(accuracies, spam_proportions)
    plt.ylabel('Accuracy')
    plt.xlabel('Spam proportion')
    plt.show()


def mean_accuracy(dataset, spam_proportion=0):
    accuracies = []
    for part_number in range(len(dataset)):
        model = MultinomialNB() if spam_proportion == 0 else MultinomialNB(fit_prior=True, class_prior=[spam_proportion, 1 - spam_proportion])
        # cross-validation splitting
        train_features, train_labels, test_features, test_labels = split_data(part_number, dataset)
        # vectorizer считает вхождения слов из тренировочных и тестовых писем
        train_word_counts = vectorizer.fit_transform(train_features)
        test_word_counts = vectorizer.transform(test_features)
        # обучаем модель и получаем результат с тестовых данных
        model.fit(train_word_counts, train_labels)
        accuracies.append(accuracy_score(test_labels, model.predict(test_word_counts)))
    return sum(accuracies)/len(accuracies)


def roc(dataset):
    messages = []
    for i in dataset:
        for message in i:
            messages.append(message)
    dataset = np.array(messages)
    features = dataset[:, 0]
    labels = dataset[:, 1]

    # Learn the vocabulary dictionary and return term-document matrix, This is equivalent to fit followed by transform, but more efficiently implemented.
    train_word_counts = vectorizer.fit_transform(features)
    # Transform documents to document-term matrix. Extract token counts out of raw text documents using the vocabulary fitted
    test_word_counts = vectorizer.transform(features)

    model = MultinomialNB()
    model.fit(train_word_counts, labels)

    # label of positive class is 0 (spam class)
    # Return probability estimates for the test vector X
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, model.predict_proba(test_word_counts)[:, 0], pos_label='0')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def zero_false_positives(dataset):
    messages = []
    for i in dataset:
        for message in i:
            messages.append(message)
    dataset = np.array(messages)

    features = dataset[:, 0]
    labels = dataset[:, 1]

    # Learn the vocabulary dictionary and return term-document matrix
    train_word_counts = vectorizer.fit_transform(features)
    # Transform documents to document-term matrix
    test_word_counts = vectorizer.transform(features)
    model = MultinomialNB(fit_prior=True, class_prior=[MIN_SPAM_PROPORTION, 1 - MIN_SPAM_PROPORTION])
    model.fit(train_word_counts, labels)
    false_positive_count = sum([1 for actual, predicted in zip(labels, model.predict(test_word_counts)) if (actual != predicted) and (actual == '1')])
    print(false_positive_count)

MIN_SPAM_PROPORTION = pow(10, -85)
dataset = prepare_data()
# Convert a collection of text documents to a matrix of token counts - Считаем частоту вхождения слов с помощью этой модели
vectorizer = CountVectorizer()
# used it to guess MIN_SPAM_PROPORTION in order to get '0' output
#zero_false_positives(dataset)
#mean_accuracy_plotter(dataset)
roc(dataset)
