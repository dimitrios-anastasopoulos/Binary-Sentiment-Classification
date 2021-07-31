import functions
import numpy as np


class ID3_Node:
    def __init__(self, parent, reviews):
        self.left = None
        self.right = None
        self.parent = parent
        self.reviews = reviews
        self.index = None
        self.category = None

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def getParent(self):
        return self.parent

    def setIndex(self, index):
        self.index = index

    def getIndex(self):
        return self.index

    def setCategory(self, category):
        self.category = category

    def getCategory(self):
        return self.category


class ID3_Tree:
    def __init__(self):
        self.head = None
        self.size = 0

    def isEmpty(self):
        return self.head is None

    def getHead(self):
        return self.head

    def getSize(self):
        return self.size

    def insert(self, reviews):
        if self.isEmpty():
            self.head = ID3_Node(None, reviews)


def ID3_train(node, review_vector_length):
    total_number_of_reviews = node.reviews.shape[0]

    vector_sum = np.zeros(review_vector_length, dtype=int)
    for review in node.reviews:
        vector_sum += review

    positive_category_reviews = vector_sum[0]
    negative_category_reviews = total_number_of_reviews - positive_category_reviews
    category_number_of_reviews = [positive_category_reviews, negative_category_reviews]

    if positive_category_reviews/total_number_of_reviews > 0.95:
        node.setCategory("Positive")
        return
    if negative_category_reviews/total_number_of_reviews > 0.95:
        node.setCategory("Negative")
        return

    ig_dictionary = {}
    h = functions.entropy(category_number_of_reviews, total_number_of_reviews)

    current_node = node
    previous_indexes_list = []
    while current_node.getParent() is not None:
        current_node = current_node.getParent()
        previous_indexes_list.append(current_node.getIndex())

    for i in range(1, review_vector_length):

        negative_category_number_of_reviews_not_containing_word = 0
        negative_category_number_of_reviews_containing_word = 0
        positive_category_number_of_reviews_not_containing_word = 0
        positive_category_number_of_reviews_containing_word = 0

        for review in node.reviews:
            if review[0] == 0:
                if review[i] == 0:
                    negative_category_number_of_reviews_not_containing_word += 1
                elif review[i] == 1:
                    negative_category_number_of_reviews_containing_word += 1
            elif review[0] == 1:
                if review[i] == 0:
                    positive_category_number_of_reviews_not_containing_word += 1
                elif review[i] == 1:
                    positive_category_number_of_reviews_containing_word += 1

        # Preparing element_option = 1:
        probability_word_contained = (negative_category_number_of_reviews_containing_word +
                                      positive_category_number_of_reviews_containing_word) / total_number_of_reviews
        category_number_of_reviews_containing_word = [probability_word_contained,
                                                      positive_category_number_of_reviews_containing_word,
                                                      negative_category_number_of_reviews_containing_word]

        # Preparing element_option = 0:
        probability_word_not_contained = 1 - probability_word_contained
        category_number_of_reviews_not_containing_word = [probability_word_not_contained,
                                                          positive_category_number_of_reviews_not_containing_word,
                                                          negative_category_number_of_reviews_not_containing_word]

        element_options = [category_number_of_reviews_containing_word, category_number_of_reviews_not_containing_word]
        ig_dictionary[i] = functions.information_gain(h, element_options)

    index = max(ig_dictionary, key=ig_dictionary.get)
    if index in previous_indexes_list:
        if positive_category_reviews > negative_category_reviews:
            node.setCategory("Positive")
        else:
            node.setCategory("Negative")
        return
    node.setIndex(index)

    left_node_reviews = []
    right_node_reviews = []

    for review in node.reviews:
        if review[index] == 1:
            left_node_reviews.append(review)
        else:
            right_node_reviews.append(review)

    node.left = ID3_Node(node, np.array(left_node_reviews))
    node.right = ID3_Node(node, np.array(right_node_reviews))

    ID3_train(node.left, review_vector_length)
    ID3_train(node.right, review_vector_length)


def ID3_evaluate(vectored_reviews, tree):

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for review in vectored_reviews:
        current_node = tree.getHead()
        while current_node.getCategory() is None:
            if review[current_node.getIndex()] == 1:
                current_node = current_node.getLeft()
            else:
                current_node = current_node.getRight()

        if current_node.getCategory() == "Positive":
            if review[0] == 1:
                true_positives += 1
            else:
                false_positives += 1
        elif current_node.getCategory() == "Negative":
            if review[0] == 1:
                false_negatives += 1
            else:
                true_negatives += 1

    print("True positives: ", true_positives)
    print("True negatives: ", true_negatives)
    print("False positives: ", false_positives)
    print("False negatives: ", false_negatives)
    return true_positives, true_negatives, false_positives, false_negatives
