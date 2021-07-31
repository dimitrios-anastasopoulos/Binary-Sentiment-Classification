import numpy as np


def naive_bayes_train(category_vectored_reviews, review_vector_length):
    """
    :param category_vectored_reviews: A python list with length equal to the total number of categories and items
                                      2d numpy arrays with shape ( len(category_reviews), len(vocabulary)+1 ) where
                                      each row represents one review of the given category.
    :param review_vector_length: An integer with the length of each review vector.
    :return: A 2d numpy array with shape ( len(categories), review_vector_length ) where each row contains a vector
             that its first item contains the probability P(C) of category C and the following items the
             probability P(X=1|C) of a review containing the word corresponding to the specific index in the vector
             given its category.
    """
    total_elements = sum([len(category) for category in category_vectored_reviews])
    category_probability_given_category_vectors = np.empty((0, review_vector_length), dtype=float)
    for category in category_vectored_reviews:
        category_probability = len(category) / total_elements
        probability_given_category_vector = np.zeros(review_vector_length, dtype=float)
        for review in category:
            probability_given_category_vector += review
        for word_prob in range(len(probability_given_category_vector)):
            probability_given_category_vector[word_prob] = (probability_given_category_vector[word_prob]+1) / \
                                                           (len(category) + len(category_vectored_reviews))
        probability_given_category_vector[0] = category_probability
        category_probability_given_category_vectors = np.vstack([category_probability_given_category_vectors,
                                                                 probability_given_category_vector])

    return category_probability_given_category_vectors


def naive_bayes_evaluate(vectored_reviews, category_probability_given_category_vectors):
    """
    :param vectored_reviews: A 2d numpy array that is returned from get_vectored_reviews() function.
    :param category_probability_given_category_vectors: A 2d numpy array that is returned from naive_bayes_train()
                                                        function.
    :return: A tuple (true_positives, true_negatives, false_positives, false_negatives) produced by the review
             classification model.
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for review in vectored_reviews:
        # Classification Process:
        category_probabilities_given_vectored_reviews = []
        for category in range(len(category_probability_given_category_vectors)):
            probability_given_vectored_review = category_probability_given_category_vectors[category][0]
            for element in range(len(category_probability_given_category_vectors[category][1:])):
                if review[element+1] == 0:
                    probability_given_vectored_review *= \
                        1 - category_probability_given_category_vectors[category][element+1]
                else:
                    probability_given_vectored_review *= \
                        category_probability_given_category_vectors[category][element+1]
            category_probabilities_given_vectored_reviews.append(probability_given_vectored_review)
        review_classification = \
            category_probabilities_given_vectored_reviews.index(max(category_probabilities_given_vectored_reviews))

        # Evaluation Process:
        if review_classification == review[0]:
            if review_classification == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if review_classification == 1:
                false_positives += 1
            else:
                false_negatives += 1

    print("True positives: ", true_positives)
    print("True negatives: ", true_negatives)
    print("False positives: ", false_positives)
    print("False negatives: ", false_negatives)
    return true_positives, true_negatives, false_positives, false_negatives
