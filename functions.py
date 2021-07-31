import re
import math
import operator
import numpy as np


def read_available_words(file):
    """
    :param file: A file pathway in .vocab format.
    :return: A dictionary (key: index, value: word) with keys representing the unique indexes of the words and values
             being the words in the file.
    """
    available_words = {}
    with open(file=file, mode="r", encoding="utf-8") as defaultVocabulary:
        counter = 0
        words = defaultVocabulary.read().splitlines()
        for word in words:
            available_words[counter] = word
            counter += 1
    return available_words


def read_reviews(file, flag=False):
    """
    :param file: A file pathway in .feat format.
    :return: A tuple of three python lists containing reviews from the .feat file.
    """
    negative_reviews = []
    positive_reviews = []
    with open(file=file, mode="r", encoding="utf-8") as labeledBow:
        reviews = labeledBow.read().splitlines()
        if flag:
            return reviews
        for review in reviews:
            if review[0:2] in "1 2 3 4 5 ":
                negative_reviews.append(review)
            else:
                positive_reviews.append(review)
    return reviews, negative_reviews, positive_reviews


def get_word_appearances(reviews, available_words):
    """
    :param reviews: A python list containing reviews from a .feat file.
    :param available_words: A dictionary (key: index, value: word) with keys representing the unique indexes of the
                            words and values being the words in our data.
    :return: A dictionary (key: index, value: number_of_appearances) with keys representing the unique indexes of the
             words and values being the number of reviews each index (word) appears in.
    """
    word_appearances = dict.fromkeys(available_words.keys(), 0)
    for review in reviews:
        words = re.findall("\\d*:\\d*", review)
        for word in words:
            index, word_counter = [int(i) for i in word.split(":")]
            word_appearances[index] += 1
    return word_appearances


def entropy(category_number_of_elements, total_number_of_elements):
    """
    :param category_number_of_elements: A python list with length equal to the total number of categories and items the
                                        number of elements in each category.
    :param total_number_of_elements: An integer that represents the total number of elements in all categories combined.
    :return: A float number of the entropy.
    """
    h = 0
    for category in category_number_of_elements:
        category_probability = category / total_number_of_elements
        h -= category_probability * math.log2(category_probability)
    return h


def entropy_given_word(category_number_of_elements_associated, total_number_of_elements_associated, h):
    """
    :param category_number_of_elements_associated: A python list with length equal to the total number of categories
                                                   and items the number of elements in each category associated with
                                                   the word.
    :param total_number_of_elements_associated: An integer that represents the total number of elements associated with
                                                the word in all categories combined.
    :param h: The entropy H(C) of variable C (category).
    :return: The entropy H(C|X=x) of variable C (category) given a value of X (word).
    """
    h_given_word = 0
    if total_number_of_elements_associated == 0:
        h_given_word = h
    else:
        for category in category_number_of_elements_associated:
            category_probability = category / total_number_of_elements_associated
            if category_probability == 0:
                return h_given_word
            h_given_word -= category_probability * math.log2(category_probability)
    return h_given_word


def information_gain(h, element_options):
    """
    :param h: The entropy H(C) of variable C (category).
    :param element_options: A python list with length equal to the total number of options for each word (in our
                            example: 0 = not contained, 1 = contained) and items python lists of the following format:
                            [P(option), category1: number_of_elements(option), category2: number_of_elements(option)],
                            where P(option) is the probability of a word having this option (e.g. be contained) and
                            number_of_elements(option) is the number of elements (in our case reviews) in a specific
                            category having this option. The list contains as many number_of_elements(option) items as
                            there are categories.
    :return: A float number with the information gain of the word.
    """
    ig = h
    for option in element_options:
        ig -= option[0] * entropy_given_word(option[1:], sum(option[1:]), h)
    return ig


def get_vocabulary(category_number_of_elements, category_word_appearances, total_word_appearances, k):
    """
    :param category_number_of_elements: A python list with length equal to the total number of categories and items the
                                        number of elements in each category.
    :param category_word_appearances: A python list with length equal to the total number of categories and items the
                                      dictionaries that are returned from the get_word_appearances() function for each
                                      category.
    :param total_word_appearances: A dictionary that is returned from the get_word_appearances() function with the
                                   total reviews of all categories as input.
    :param k: The number of indexes (words) that we want to include in our vocabulary.
    :return: A dictionary (key: index, value: information_gain) with keys representing the unique indexes of the words
             and values being the return values of the information_gain() function for each specific index (word).
    """
    vocabulary = {}
    h = entropy(category_number_of_elements, sum(category_number_of_elements))

    for word_index in total_word_appearances:
        # Preparing element_option = 1:
        probability_word_contained = total_word_appearances[word_index] / sum(category_number_of_elements)
        category_number_of_elements_containing_word = [category[word_index] for category in category_word_appearances]
        category_number_of_elements_containing_word.insert(0, probability_word_contained)
        # Preparing element_option = 0:
        probability_word_not_contained = 1 - probability_word_contained
        category_number_of_elements_not_containing_word = [category_number_of_elements[category] -
                                                           category_word_appearances[category][word_index]
                                                           for category in range(len(category_word_appearances))]
        category_number_of_elements_not_containing_word.insert(0, probability_word_not_contained)

        element_options = [category_number_of_elements_containing_word, category_number_of_elements_not_containing_word]

        vocabulary[word_index] = information_gain(h, element_options)

    vocabulary = dict(sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)[:k])

    return vocabulary


def get_reviews_vectors(reviews, vocabulary):
    """
    :param reviews: A python list containing reviews from a .feat file.
    :param vocabulary: A dictionary (key: index, value: information_gain) with keys representing the unique indexes of
                       the words and values being the return values of the information_gain() function for each specific
                       index (word).
    :return: A python array with shape ( len(reviews), len(vocabulary)+1 ) where each row represents one review.
             Every review is a binary numpy array. The first item indicates the category of the review and the
             following items whether or not this review contains a specific index (word) from the vocabulary.
    """
    reviews_vectors = []
    for review in reviews:
        if review[0:2] in "1 2 3 4 5 ":
            review_vector = [0]
        else:
            review_vector = [1]

        review_dict = {}
        for key in vocabulary.keys():
            review_dict[key] = 0
        words = re.findall("\\d*:\\d*", review)
        for word in words:
            index, word_counter = [int(i) for i in word.split(":")]
            if index in vocabulary.keys():
                review_dict[index] = 1
        for value in review_dict.values():
            review_vector.append(value)
        reviews_vectors.append(review_vector)
    return reviews_vectors


def get_vectored_reviews_np(reviews, vocabulary):
    """
    :param reviews: A python list containing reviews from a .feat file.
    :param vocabulary: A dictionary (key: index, value: information_gain) with keys representing the unique indexes of
                       the words and values being the return values of the information_gain() function for each specific
                       index (word).
    :return: A 2d numpy array with shape ( len(reviews), len(vocabulary)+1 ) where each row represents one review.
             Every review is a binary numpy array. The first item indicates the category of the review and the
             following items whether or not this review contains a specific index (word) from the vocabulary.
    """
    vectored_reviews = np.empty((0, len(vocabulary)+1), dtype=int)
    for review in reviews:
        if review[0:2] in "1 2 3 4 5 ":
            review_vector = np.array([[0]])
        else:
            review_vector = np.array([[1]])

        review_dict = {}
        for key in vocabulary.keys():
            review_dict[key] = 0
        words = re.findall("\\d*:\\d*", review)
        for word in words:
            index, word_counter = [int(i) for i in word.split(":")]
            if index in vocabulary.keys():
                review_dict[index] = 1
        for value in review_dict.values():
            review_vector = np.append(review_vector, np.array([[value]]), axis=1)
        vectored_reviews = np.append(vectored_reviews, review_vector, axis=0)
    return vectored_reviews
