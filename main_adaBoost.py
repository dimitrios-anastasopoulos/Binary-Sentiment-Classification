import functions
import adaBoost
import metrics
import numpy as np


def run_adaboost():
    while True:
        choice = input("\nAdaBoost Algorithm Menu"
                       "\n1. Train & Evaluate"
                       "\n2. Calculate & Plot Learning Curve"
                       "\n3. Plot Learning Curve"
                       "\n4. Test Parameters on Dev Data"
                       "\n0. Exit"
                       "\nSelect: ")

        if choice == '1':
            print("\nTraining with AdaBoost using optimal parameter xxx...")
            available_words = functions.read_available_words("imdb.vocab")
            reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
            train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics.split_train_dev(
                negative_reviews, positive_reviews)

            word_appearances = functions.get_word_appearances(train_reviews, available_words)
            negative_word_appearances = functions.get_word_appearances(train_negative_reviews, available_words)
            positive_word_appearances = functions.get_word_appearances(train_positive_reviews, available_words)

            vocabulary = functions.get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)],
                                                  [negative_word_appearances, positive_word_appearances],
                                                  word_appearances, 300)

            train_vectored_reviews = np.array(functions.get_reviews_vectors(train_reviews, vocabulary))
            train_reviews_values = train_vectored_reviews[:, 0]

            classifiers = adaBoost.train(train_vectored_reviews[:, 1:], train_reviews_values, len(vocabulary), 600)

            test_reviews, test_negative_reviews, test_positive_reviews = functions.read_reviews("test//labeledBow.feat")

            test_vectored_reviews = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))
            test_reviews_values = test_vectored_reviews[:, 0]

            print("\nAlgorithm Evaluation")
            predictions = adaBoost.predict(test_vectored_reviews[:, 1:], classifiers)
            true_positives, true_negatives, false_positives, false_negatives = adaBoost.evaluate(test_reviews_values,
                                                                                                 predictions)
            accuracy = metrics.get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
            print("Algorithm accuracy: {0}\n".format(accuracy))

        elif choice == '2':
            available_words = functions.read_available_words("imdb.vocab")
            reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
            train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics.split_train_dev(
                negative_reviews, positive_reviews)
            test_reviews, test_negative_reviews, test_positive_reviews = functions.read_reviews(
                "test//labeledBow.feat")
            x_values, train_y_values, test_y_values = metrics.get_learning_curve_points(train_negative_reviews,
                                                                                        train_positive_reviews,
                                                                                        test_reviews,
                                                                                        available_words, "adaBoost")
            print(x_values, train_y_values, test_y_values)
            metrics.plot_learning_curve(x_values, train_y_values, test_y_values)

        elif choice == '3':
            with open("..//learning_curves//adaBoost_learning_curve_points.txt", mode='r') as file:
                data = file.read().splitlines()
                list_data = []
                for d in data:
                    d = d.split(',')
                    list_data.append(d)
                data = np.array(list_data[1:]).astype(np.float)

            x_values = data[1:, 0]
            train_y_values = data[1:, 1]
            test_y_values = data[1:, 2]

            metrics.plot_learning_curve(x_values, train_y_values, test_y_values)

        elif choice == '4':
            parameter_words = input("\nPlease give the parameters with which you want to evaluate the Dev Data"
                                    "\nParameter for number of words used: ")
            parameter_classifiers = input("\nParameter for number of classifiers used: ")

            try:
                if isinstance(int(parameter_words), int) and isinstance(int(parameter_classifiers), int):
                    print(
                        "Training with AdaBoost using {0} words and {1} classifiers for the evaluation process".format(
                            parameter_words, parameter_classifiers))
                    available_words = functions.read_available_words("imdb.vocab")
                    reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
                    train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics.split_train_dev(
                        negative_reviews, positive_reviews)

                    word_appearances = functions.get_word_appearances(train_reviews, available_words)
                    negative_word_appearances = functions.get_word_appearances(train_negative_reviews, available_words)
                    positive_word_appearances = functions.get_word_appearances(train_positive_reviews, available_words)

                    vocabulary = functions.get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)],
                                                          [negative_word_appearances, positive_word_appearances],
                                                          word_appearances, int(parameter_words))

                    train_vectored_reviews = np.array(functions.get_reviews_vectors(train_reviews, vocabulary))
                    train_reviews_values = train_vectored_reviews[:, 0]

                    classifiers = adaBoost.train(train_vectored_reviews[:, 1:], train_reviews_values, len(vocabulary),
                                                 int(parameter_classifiers))

                    dev_vectored_reviews = np.array(functions.get_reviews_vectors(dev_reviews, vocabulary))
                    dev_reviews_values = dev_vectored_reviews[:, 0]

                    print("\nAlgorithm Evaluation")
                    predictions = adaBoost.predict(dev_vectored_reviews[:, 1:], classifiers)
                    true_positives, true_negatives, false_positives, false_negatives = adaBoost.evaluate(
                        dev_reviews_values, predictions)
                    accuracy = metrics.get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
                    print("Algorithm accuracy: {0}\n".format(accuracy))

            except ValueError:
                print("\nParameter must be type integer!\n")

        elif choice == '0':
            break

        else:
            print("\nInvalid input\n")
