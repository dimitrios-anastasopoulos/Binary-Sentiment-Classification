import functions
import naiveBayes
import metrics
import numpy as np


def run_naive_bayes():
    while True:
        choice = input("\nNaive Bayes Algorithm Menu"
                       "\n1. Train & Evaluate"
                       "\n2. Calculate & Plot Learning Curve"
                       "\n3. Plot Learning Curve"
                       "\n4. Test Parameter on Dev Data"
                       "\n0. Exit"
                       "\nSelect: ")

        if choice == "1":
            print("\nTraining with Naive Bayes using optimal parameter 1200...")
            available_words = functions.read_available_words("imdb.vocab")
            reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
            train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics.split_train_dev(
                negative_reviews, positive_reviews)

            word_appearances = functions.get_word_appearances(train_reviews, available_words)
            negative_word_appearances = functions.get_word_appearances(train_negative_reviews, available_words)
            positive_word_appearances = functions.get_word_appearances(train_positive_reviews, available_words)

            vocabulary = functions.get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)],
                                                  [negative_word_appearances, positive_word_appearances],
                                                  word_appearances, 1200)

            negative_vectored_reviews = np.array(functions.get_reviews_vectors(train_negative_reviews, vocabulary))
            positive_vectored_reviews = np.array(functions.get_reviews_vectors(train_positive_reviews, vocabulary))

            category_probability_given_category_vectors = naiveBayes.naive_bayes_train(
                [negative_vectored_reviews, positive_vectored_reviews], len(vocabulary) + 1)

            test_reviews, test_negative_reviews, test_positive_reviews = functions.read_reviews("test//labeledBow.feat")

            test_vectored_reviews = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))

            print("\nAlgorithm Evaluation")
            true_positives, true_negatives, false_positives, false_negatives = naiveBayes.naive_bayes_evaluate(
                test_vectored_reviews, category_probability_given_category_vectors)
            accuracy = metrics.get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
            print("Algorithm accuracy: {0}\n".format(accuracy))

        elif choice == "2":
            warning = input("\nWarning! This is going to take between 1-2 hours to finish. Do you wish to proceed? (y/n)"
                            "\nSelect : ")
            if warning == "y":
                available_words = functions.read_available_words("imdb.vocab")
                reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
                train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics.split_train_dev(
                    negative_reviews, positive_reviews)
                test_reviews, test_negative_reviews, test_positive_reviews = functions.read_reviews(
                    "test//labeledBow.feat")
                x_values, train_y_values, test_y_values = metrics.get_learning_curve_points(train_negative_reviews,
                                                                                            train_positive_reviews,
                                                                                            test_reviews,
                                                                                            available_words,
                                                                                            "naiveBayes")
                metrics.plot_learning_curve(x_values, train_y_values, test_y_values)

        elif choice == "3":
            with open("..//learning_curves//naiveBayes_learning_curve_points.txt", mode='r') as file:
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

        elif choice == "4":
            parameter = input("\nPlease give the parameter with which you want to evaluate the Dev Data"
                              "\nParameter: ")

            try:
                if isinstance(int(parameter), int):

                    print("Training with Naive Bayes using parameter {0}".format(parameter))
                    available_words = functions.read_available_words("imdb.vocab")
                    reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
                    train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics.split_train_dev(
                        negative_reviews, positive_reviews)

                    word_appearances = functions.get_word_appearances(train_reviews, available_words)
                    negative_word_appearances = functions.get_word_appearances(train_negative_reviews, available_words)
                    positive_word_appearances = functions.get_word_appearances(train_positive_reviews, available_words)

                    vocabulary = functions.get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)],
                                                          [negative_word_appearances, positive_word_appearances],
                                                          word_appearances, int(parameter))

                    negative_vectored_reviews = np.array(
                        functions.get_reviews_vectors(train_negative_reviews, vocabulary))
                    positive_vectored_reviews = np.array(
                        functions.get_reviews_vectors(train_positive_reviews, vocabulary))

                    category_probability_given_category_vectors = naiveBayes.naive_bayes_train(
                        [negative_vectored_reviews, positive_vectored_reviews], len(vocabulary) + 1)

                    dev_vectored_reviews = np.array(functions.get_reviews_vectors(dev_reviews, vocabulary))

                    print("\nAlgorithm Evaluation")
                    true_positives, true_negatives, false_positives, false_negatives = naiveBayes.naive_bayes_evaluate(
                        dev_vectored_reviews, category_probability_given_category_vectors)
                    accuracy = metrics.get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
                    print("Algorithm accuracy on Dev Data with parameter {0}: {1}\n".format(parameter, accuracy))

            except ValueError:
                print("\nParameter must be type integer!\n")

        elif choice == "0":
            break

        else:
            print("\nInvalid Input\n")
