import functions
import naiveBayes
import ID3
import adaBoost
import numpy as np
import matplotlib.pyplot as plt


def split_train_dev(negative_reviews, positive_reviews):

    pos_percent_95 = round(len(positive_reviews) * 0.95)
    neg_percent_95 = round(len(negative_reviews) * 0.95)

    train_positive_reviews = positive_reviews[:pos_percent_95]
    train_negative_reviews = negative_reviews[:neg_percent_95]
    train_reviews = train_positive_reviews + train_negative_reviews

    dev_reviews = negative_reviews[neg_percent_95:] + positive_reviews[pos_percent_95:]

    return train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews


def get_accuracy(true_positives, true_negatives, false_positives, false_negatives):
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    return accuracy


def get_error(true_positives, true_negatives, false_positives, false_negatives):
    error = (false_positives + false_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    return error


def get_learning_curve_points(train_negative_reviews, train_positive_reviews, test_reviews, available_words, algorithm):
    x_values = []
    train_y_values = []
    test_y_values = []

    for value in np.linspace(0.05, 1.0, num=20):
        x_values.append(value)

        print("\nPercentage of training data used: {0}%".format(int(round(value*100))))

        used_train_negative_reviews = train_negative_reviews[:int(len(train_negative_reviews)*value)]
        used_train_positive_reviews = train_positive_reviews[:int(len(train_positive_reviews)*value)]
        used_train_reviews = used_train_negative_reviews + used_train_positive_reviews

        word_appearances = functions.get_word_appearances(used_train_reviews, available_words)
        negative_word_appearances = functions.get_word_appearances(used_train_negative_reviews, available_words)
        positive_word_appearances = functions.get_word_appearances(used_train_positive_reviews, available_words)

        if algorithm == "naiveBayes":
            vocabulary = functions.get_vocabulary([len(used_train_negative_reviews), len(used_train_positive_reviews)],
                                                  [negative_word_appearances, positive_word_appearances],
                                                  word_appearances, 1200)

            negative_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_negative_reviews, vocabulary))
            positive_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_positive_reviews, vocabulary))

            category_probability_given_category_vectors = naiveBayes.naive_bayes_train(
                [negative_vectored_reviews, positive_vectored_reviews], len(vocabulary) + 1)

            train_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_reviews, vocabulary))

            print("\nAlgorithm Evaluation for Train Data")
            train_true_positives, train_true_negatives, train_false_positives, train_false_negatives = naiveBayes.naive_bayes_evaluate(
                train_vectored_reviews, category_probability_given_category_vectors)
            train_accuracy = get_accuracy(train_true_positives, train_true_negatives, train_false_positives,
                                          train_false_negatives)
            train_y_values.append(train_accuracy)

            test_vectored_reviews = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))

            print("\nAlgorithm Evaluation for Test Data")
            test_true_positives, test_true_negatives, test_false_positives, test_false_negatives = naiveBayes.naive_bayes_evaluate(
                test_vectored_reviews, category_probability_given_category_vectors)
            test_accuracy = get_accuracy(test_true_positives, test_true_negatives, test_false_positives,
                                         test_false_negatives)
            test_y_values.append(test_accuracy)

        elif algorithm == "ID3":
            vocabulary = functions.get_vocabulary([len(used_train_negative_reviews), len(used_train_positive_reviews)],
                                                  [negative_word_appearances, positive_word_appearances],
                                                  word_appearances, 22)

            train_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_reviews, vocabulary))

            tree = ID3.ID3_Tree()
            tree.insert(train_vectored_reviews)
            ID3.ID3_train(tree.getHead(), len(vocabulary) + 1)

            print("\nAlgorithm Evaluation for Train Data")
            train_true_positives, train_true_negatives, train_false_positives, train_false_negatives = ID3.ID3_evaluate(
                train_vectored_reviews, tree)
            train_accuracy = get_accuracy(train_true_positives, train_true_negatives, train_false_positives,
                                          train_false_negatives)
            train_y_values.append(train_accuracy)

            test_vectored_reviews = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))

            print("\nAlgorithm Evaluation for Test Data")
            test_true_positives, test_true_negatives, test_false_positives, test_false_negatives = ID3.ID3_evaluate(
                test_vectored_reviews, tree)
            test_accuracy = get_accuracy(test_true_positives, test_true_negatives, test_false_positives,
                                         test_false_negatives)
            test_y_values.append(test_accuracy)

        elif algorithm == "adaBoost":
            vocabulary = functions.get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)],
                                                  [negative_word_appearances, positive_word_appearances],
                                                  word_appearances, 300)

            train_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_reviews, vocabulary))
            train_reviews_values = train_vectored_reviews[:, 0]

            classifiers = adaBoost.train(train_vectored_reviews[:, 1:], train_reviews_values, len(vocabulary), 600)

            print("\nAlgorithm Evaluation for Train Data")
            predictions = adaBoost.predict(train_vectored_reviews[:, 1:], classifiers)
            true_positives, true_negatives, false_positives, false_negatives = adaBoost.evaluate(train_reviews_values,
                                                                                                 predictions)
            train_accuracy = get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
            train_y_values.append(train_accuracy)

            test_vectored_reviews = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))
            test_reviews_values = test_vectored_reviews[:, 0]

            print("\nAlgorithm Evaluation for Test Data")
            predictions = adaBoost.predict(test_vectored_reviews[:, 1:], classifiers)
            true_positives, true_negatives, false_positives, false_negatives = adaBoost.evaluate(test_reviews_values,
                                                                                                 predictions)
            test_accuracy = get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
            test_y_values.append(test_accuracy)

    return x_values, train_y_values, test_y_values


def plot_learning_curve(x_values, train_y_values, test_y_values):
    fig = plt.figure(figsize=(10, 6))

    fig.suptitle("Learning Curves")

    acc_chart = fig.add_subplot(121)
    acc_chart.plot(x_values, train_y_values, color="k")
    acc_chart.plot(x_values, test_y_values, color="r")
    acc_chart.set_xlabel("Percentage of data used.")
    acc_chart.set_ylabel("Accuracy")
    acc_chart.legend(["Train Data", "Test Data"])

    error_chart = fig.add_subplot(122)
    error_chart.plot(x_values, [1-acc for acc in train_y_values], color="k")
    error_chart.plot(x_values, [1-acc for acc in test_y_values], color="r")
    error_chart.set_xlabel("Percentage of data used.")
    error_chart.set_ylabel("Error")
    error_chart.legend(["Train Data", "Test Data"])

    plt.show()
