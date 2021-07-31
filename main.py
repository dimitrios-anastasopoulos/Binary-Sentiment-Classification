from main_adaBoost import run_adaboost
from main_ID3 import run_ID3
from main_naiveBayes import run_naive_bayes
import os


if __name__ == '__main__':

    os.chdir("aclImdb")

    while True:
        print("\nMachine Learning - Binary Sentiment Review Classification")

        method = input("\nInsert Training Method (1,2,3) to implement:11"
                       "\n1. Naive Bayes"
                       "\n2. ID3"
                       "\n3. AdaBoost"
                       "\nSelect: ")

        if method == '1':
            run_naive_bayes()
        elif method == '2':
            run_ID3()
        elif method == '3':
            run_adaboost()
        else:
            print("\nError: Try using the numbers '1', '2', '3' to navigate through the menu.\n")

        repeat = input("\nDo you want to try some other algorithm?"
                       "\n0: No"
                       "\n1: Yes"
                       "\nSelect: ")

        if repeat == '1':
            continue
        else:
            print("\nProgram Shutdown...\n")
            break
