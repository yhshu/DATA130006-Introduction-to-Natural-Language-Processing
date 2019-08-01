import ast
import math
from collections import Counter

import nltk
# THE CONSTANTS
from nltk.corpus import reuters

from util import rchop, lchop

download_needed = False  # default: True, when the corpus is used for the first time, downloading is necessary.

suffix_list = ["'d", "'flight", "'ll", "'re", "'s", "'ve"]

# file path
test_data_path = "testdata.txt"
result_file_path = "result.txt"
ans_file_path = "ans.txt"
vocabulary_path = "vocab.txt"

EDIT_TYPE_INSERTION = 0
EDIT_TYPE_DELETION = 1
EDIT_TYPE_SUBSTITUTION = 2
EDIT_TYPE_TRANSPOSITION = 3


class SpellingCorrector:
    # member fields

    word_list = []  # splice all the words in the corpus
    vocab_list = []

    # confusion matrix
    add_mat = {}
    sub_mat = {}
    del_mat = {}
    rev_mat = {}

    # n-gram count
    unigram = {}
    bigram = {}
    trigram = {}
    four_gram = {}
    five_gram = {}

    def __init__(self):
        self.word_list = self.load_corpus("reuters")
        # load the corpus to create the word list
        # note that the network is needed to download the corpus

        self.count_ngrams()  # count the n-grams
        self.load_confusion_matrix()  # read the confusion matrix from files
        self.load_vocabulary()  # read the vocabulary from a file

    def load_corpus(self, corpus_name):
        if corpus_name == "reuters":
            return self.load_reuters_corpus()

    def load_reuters_corpus(self):
        """
        Get Reuters Corpus from nltk package
        :returns a list of words to splicing all the files in the corpus
        """

        if download_needed:
            nltk.download('reuters')
        else:
            print(
                "[INFO] The local corpus is used, if you have not already downloaded the corpus, set the download_needed = True after connecting to the Internet")
        reuters_fileids = reuters.fileids()
        print("[INFO] The length of reuters corpus is " + str(len(reuters_fileids)))
        reuters_categories = reuters.categories()
        print("[INFO] The length of reuters categories is " + str(len(reuters_categories)))
        print("[INFO] Checking if the corpus file has been obtained...")
        print(reuters_fileids[0:5])  # check if the files has been downloaded

        word_list = []
        for file_id in reuters_fileids:
            file_word = nltk.corpus.reuters.words(file_id)  # words in this file
            file_word_lowercase = [w.lower() for w in file_word]  # convert words to lowercase
            word_list.extend(file_word_lowercase)
        print("[INFO] Load Reuters corpus completed")
        return word_list

    def load_vocabulary(self):
        """
        load the vocabulary from file
        """
        vocab_file = open(vocabulary_path, "r")
        self.vocab_list = vocab_file.read().split("\n")
        vocab_file.close()
        print("[INFO] Reading vocabulary...")
        print(self.vocab_list[0:15])

    def edit_distance(self, str1, str2, m, n):
        """
        A Naive recursive Python program to find minimum number operations to convert str1 to str2
        """

        # If first string is empty, the only option is to
        # insert all characters of second string into first
        if m == 0:
            return n

            # If second string is empty, the only option is to
        # remove all characters of first string
        if n == 0:
            return m

            # If last characters of two strings are same, nothing
        # much to do. Ignore last characters and get count for
        # remaining strings.
        if str1[m - 1] == str2[n - 1]:
            return self.edit_distance(str1, str2, m - 1, n - 1)

            # If last characters are not same, consider all three
        # operations on last character of first string, recursively
        # compute minimum cost for all three operations and take
        # minimum of three values.
        return 1 + min(self.edit_distance(str1, str2, m, n - 1),  # Insert
                       self.edit_distance(str1, str2, m - 1, n),  # Remove
                       self.edit_distance(str1, str2, m - 1, n - 1)  # Replace
                       )

    def Damerau_Levenshtein_edit_distance(self, str1, str2):
        """
        Calculate Damerau-Levenshtein Edit Distance for two string

        Reference:
        https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance

        :param str1: string 1
        :param str2: string 2
        :return: the Damerau-Levenshtein Edit Distance between str1 and str2
        """
        str1 = '@' + str1
        str2 = '@' + str2
        len1 = len(str1)
        len2 = len(str2)
        dis = [[0] * len2 for _ in range(len1)]
        for i in range(len1):
            for j in range(len2):
                dis[i][0] = i
                dis[0][j] = j

        for i in range(len1):
            for j in range(len2):
                if i == 0 or j == 0:
                    continue  # dis[0][0] = 0

                t = [0] * 4
                t[0] = dis[i - 1][j] + 1
                t[1] = dis[i][j - 1] + 1
                if str1[i] != str2[j]:
                    t[2] = dis[i - 1][j - 1] + 1
                else:
                    t[2] = dis[i - 1][j - 1]
                if str1[i] == str2[j - 1] and str1[i - 1] == str2[j]:  # transposition of two adjacent characters
                    t[3] = dis[i - 1][j - 1] - 1
                if t[3] != 0:
                    dis[i][j] = min(t[0:4])
                else:
                    dis[i][j] = min(t[0:3])
        return dis[len1 - 1][len2 - 1]

    def get_candidates(self, word):
        """
        Damerau-Levenshtein edit distance is used to generate a candidate set of this word.
        :param word: source word used to generate a candidate set
        :return: the candidate set of this word
        """
        candidates = dict()
        for word_list_item in self.word_list:
            edit_distance = self.Damerau_Levenshtein_edit_distance(word, word_list_item)
            if edit_distance <= 1:
                candidates[word_list_item] = edit_distance
        return sorted(candidates, key=candidates.get, reverse=False)

    def edit_type(self, candidate, word):
        """
        Calculate edit type for a single edit error
        :param candidate: a candidate word in the candidate set
        :param word: one word in an original sentence
        :return: edit type
        """
        edit = [False] * 4
        correct = ""
        error = ""
        replaced = ''
        replacer = ''
        for i in range(min([len(word), len(candidate)]) - 1):
            if candidate[0:i + 1] != word[0:i + 1]:  # inconsistency in the first (i + 1) characters of the two strings
                if candidate[i:] == word[i - 1:]:
                    edit[1] = True  # deletion
                    correct = candidate[i - 1]  # candidate[i - 1] is deleted and we get word
                    error = ''
                    replaced = candidate[i - 2]
                    replacer = candidate[i - 2] + candidate[i - 1]
                    break
                elif candidate[i:] == word[i + 1:]:
                    edit[0] = True  # insertion
                    correct = ''
                    error = word[i]  # word[i] is redundant
                    if i == 0:
                        replacer = '@'
                        replaced = '@' + error
                    else:
                        replacer = word[i - 1]
                        replaced = word[i - 1] + error
                    break
                if candidate[i + 1:] == word[i + 1:]:
                    edit[2] = True  # substitution
                    correct = candidate[i]
                    error = word[i]
                    replaced = error
                    replacer = correct
                    break
                if candidate[i] == word[i + 1] and candidate[i + 2:] == word[i + 2:]:
                    edit[3] = True  # transposition
                    correct = candidate[i] + candidate[i + 1]
                    error = word[i] + word[i + 1]
                    replaced = error
                    replacer = correct
                    break

        # string inversion
        candidate = candidate[::-1]
        word = word[::-1]

        for i in range(min([len(word), len(candidate)]) - 1):
            if candidate[0:i + 1] != word[0:i + 1]:
                if candidate[i:] == word[i - 1:]:
                    edit[1] = True
                    correct = candidate[i - 1]
                    error = ''
                    replaced = candidate[i - 2]
                    replacer = candidate[i - 2] + candidate[i - 1]
                    break
                elif candidate[i:] == word[i + 1:]:
                    correct = ''
                    error = word[i]
                    if i == 0:
                        replacer = '@'
                        replaced = '@' + error
                    else:
                        replacer = word[i - 1]
                        replaced = word[i - 1] + error
                    edit[0] = True
                    break
                if candidate[i + 1:] == word[i + 1:]:
                    edit[2] = True
                    correct = candidate[i]
                    error = word[i]
                    replaced = error
                    replacer = correct
                    break
                if candidate[i] == word[i + 1] and candidate[i + 2:] == word[i + 2:]:
                    edit[3] = True
                    correct = candidate[i] + candidate[i + 1]
                    error = word[i] + word[i + 1]
                    replaced = error
                    replacer = correct
                    break

        if word == candidate:
            return "None", '', '', '', ''
        if edit[0]:
            return EDIT_TYPE_INSERTION, correct, error, replaced, replacer
        elif edit[1]:
            return EDIT_TYPE_DELETION, correct, error, replaced, replacer
        elif edit[2]:
            return EDIT_TYPE_SUBSTITUTION, correct, error, replaced, replacer
        elif edit[3]:
            return EDIT_TYPE_TRANSPOSITION, correct, error, replaced, replacer

    def load_data_file(self, file_path):
        """
        load data from a file
        :param file_path: the file path
        :return: Python data structure
        """
        file = open(file_path, "r")
        data = file.read()
        file.close()
        return ast.literal_eval(data)

    def load_confusion_matrix(self):
        """
        load confusion matrix from files
        the confusion matrix comes from the paper `A Spelling Correction Program Based on a Noisy Channel Model`
        """
        self.add_mat = self.load_data_file('confusion_matrix/add.txt')
        self.sub_mat = self.load_data_file('confusion_matrix/sub.txt')
        self.del_mat = self.load_data_file('confusion_matrix/del.txt')
        self.rev_mat = self.load_data_file('confusion_matrix/rev.txt')
        print(
            "[INFO] Load confusion matrix completed: \n" +
            "[INFO] confusion matrix add_mat: " + str(self.add_mat) + "\n" +
            "[INFO] confusion matrix sub_mat: " + str(self.sub_mat) + "\n" +
            "[INFO] confusion matrix del_mat: " + str(self.del_mat) + "\n" +
            "[INFO] confusion matrix rev_mat: " + str(self.rev_mat))

    def channel_model(self, str1, str2, edit_type):
        """
        Calculate channel model probability for errors
        :param str1: string 1
        :param str2: string 2
        :param edit_type: the edit type including insertion, deletion, substitution and transposition
        :return:
        """
        corpus = " ".join(self.word_list)  # use spaces to join all the elements in the list
        string = str1.lower() + str2.lower()
        if edit_type == EDIT_TYPE_INSERTION:
            if str1 == "@":
                if corpus.count(" " + str2):
                    return 0
                return self.add_mat[string] / corpus.count(" " + str2)
            else:
                if corpus.count(str1):
                    return 0
                return self.add_mat[string] / corpus.count(str1)
        if corpus.count(string):
            return 0
        if edit_type == EDIT_TYPE_DELETION:
            return self.del_mat[string] / corpus.count(string)
        if edit_type == EDIT_TYPE_SUBSTITUTION:
            return self.sub_mat[string] / corpus.count(string)
        if edit_type == EDIT_TYPE_TRANSPOSITION:
            return self.rev_mat[string] / corpus.count(string)

    def count_ngrams(self):
        self.unigram = self.count_unigram(self.word_list)
        self.bigram = self.count_bigram(self.word_list)
        # self.trigram = self.count_trigram(self.word_list)
        # self.four_gram = self.count_four_gram(self.word_list)
        # self.five_gram = self.count_five_gram(self.word_list)

    def count_unigram(self, word_list):
        return Counter(word_list)

    def count_bigram(self, word_list):
        bigram = []
        for i in range(len(word_list)):
            if i >= len(word_list) - 1:  # the last one
                break
            bigram.append(word_list[i] + " " + word_list[i + 1])
        return Counter(bigram)

    def count_trigram(self, word_list):
        trigram = []
        for i in range(len(word_list)):
            if i >= len(word_list) - 2:
                break
            trigram.append(word_list[i] + " " + word_list[i + 1] + " " + word_list[i + 2])
        return Counter(trigram)

    def count_four_gram(self, word_list):
        four_gram = []
        for i in range(len(word_list)):
            if i >= len(word_list) - 3:
                break
            four_gram.append(word_list[i] + " " + word_list[i + 1] + " " + word_list[i + 2] + " " + word_list[i + 3])
        return Counter(four_gram)

    def count_five_gram(self, word_list):
        five_gram = []
        for i in range(len(word_list)):
            if i >= len(word_list) - 4:
                break
            five_gram.append(
                word_list[i] + " " + word_list[i + 1] + " " + word_list[i + 2] + " " + word_list[i + 3] + " " +
                word_list[i + 4])
        return Counter(five_gram)

    def n_gram_MLE(self, word, words, gram_type=1):
        """
        Calculate the Maximum Likelihood Probability of n-grams with Laplace smoothing

        Reference:
        - http://lintool.github.io/UMD-courses/CMSC723-2009-Fall/session9-slides.pdf
        """

        if gram_type == 1:
            return math.log((self.unigram[word] + 1) / (len(self.word_list) + len(self.unigram)))
        elif gram_type == 2:
            return math.log(self.bigram[words] + 1) / (self.unigram[word] + len(self.unigram))
        elif gram_type == 3:
            return math.log(self.trigram[words] + 1) / (self.bigram[word] + len(self.unigram))
        elif gram_type == 4:
            return math.log(self.four_gram[words] + 1) / (self.trigram[word] + len(self.unigram))
        elif gram_type == 5:
            return math.log(self.five_gram[words] + 1) / (self.four_gram[word] + len(self.unigram))

    def sentence_probability(self, sentence, ngram_type=1):
        sentence_word_list = sentence.lower().split()
        prob = 0  # sentence probability
        if ngram_type == 1:
            for i in range(len(sentence_word_list)):
                prob = prob + self.sentence_probability(sentence_word_list[i])

        if ngram_type == 2:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 1:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i],
                                              sentence_word_list[i] + " " + sentence_word_list[i + 1], 2)

        if ngram_type == 3:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 2:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i] + ' ' + sentence_word_list[i + 1],
                                              sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2], 3)

        if ngram_type == 4:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 3:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2],
                                              sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2] + sentence_word_list[i + 3], 4)

        if ngram_type == 5:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 4:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2] + sentence_word_list[i + 3],
                                              sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2] + sentence_word_list[i + 3] +
                                              sentence_word_list[i + 4], 5)

        return prob

    def word_list_clean(self, word_list):
        """
        There may be an empty element in the sentence list, and the last element may contain a period and "\n"
        :param word_list: A list of words to be cleaned, usually consisting of words in one or several sentences
        :return: the cleaned word list
        """
        word_list = [word for word in word_list if word != '']  # remove empty string in the sentence list
        for i in range(len(word_list)):
            word_list[i] = word_list[i].strip("\n")  # remove the "\n"
        return word_list

    def word_clean(self, word):
        word_ori = word
        if word not in self.vocab_list:  # if the word is not in the vocabulary
            word = word.strip(",.")  # delete punctuation, such as periods, commas
        for i in range(len(suffix_list)):
            (match, string) = rchop(word, suffix_list[i])
            (_, suffix) = lchop(word_ori, word)
            if match:
                return string, suffix

        (_, suffix) = lchop(word_ori, word)
        return word, suffix


def sentence_spelling_correction(test_data_line):
    test_data_line = test_data_line.split("\t")  # split the test_data_line according to the metadata
    sentence_id = test_data_line[0]  # sentence id
    n_error = int(test_data_line[1])  # the error number in this sentence
    sentence = test_data_line[2]  # the test sentence
    sentence = sentence.split(" ")  # split words in one sentence
    sentence = spelling_corrector.word_list_clean(sentence)

    result_sentence = ""  # The sentence after the correction

    # First, we determine the number of non-word errors by comparing the vocabulary.
    # If the number of errors is still greater than 0 except non-word errors, we begin to correct real word errors
    n_non_word_error = 0  # count the number of non-word errors
    non_word_index_list = []
    for word_i, word in enumerate(sentence):
        # note that comparing original sentence (with uppercase) with the vocabulary
        if word not in spelling_corrector.vocab_list and \
                spelling_corrector.word_clean(word)[0] not in spelling_corrector.vocab_list:
            # it's a non-word error
            n_non_word_error = n_non_word_error + 1
            non_word_index_list.append(word_i)

    print("[INFO] sentence id: " + str(sentence_id) + ", n_error: " + str(n_error) +
          ", non-word error: " + str(n_non_word_error))
    print("[INFO] original sentence: " + str(sentence))

    for word_i, word in enumerate(sentence):  # for each word in the original sentence
        if word_i not in non_word_index_list and n_non_word_error >= n_error:
            # it's correct
            result_sentence = result_sentence + word + " "
            continue

        # it's non-word error or real word error
        word_cleaned, word_suffix = spelling_corrector.word_clean(word)
        word_candidates = spelling_corrector.get_candidates(word_cleaned)  # get a candidate set for the word
        if word_cleaned in word_candidates:
            result_sentence = result_sentence + word + ' '
            continue

        channel_dict = dict()
        prob_dict = dict()

        for candidate_item in word_candidates:
            edit = spelling_corrector.edit_type(candidate_item, word_cleaned)
            if edit is None:
                continue
            if edit[0] == EDIT_TYPE_INSERTION:
                channel_dict[candidate_item] = spelling_corrector.channel_model(str1=edit[3][0], str2=edit[3][1],
                                                                                edit_type=EDIT_TYPE_INSERTION)
            if edit[0] == EDIT_TYPE_DELETION:
                channel_dict[candidate_item] = spelling_corrector.channel_model(str1=edit[4][0], str2=edit[4][1],
                                                                                edit_type=EDIT_TYPE_DELETION)
            if edit[0] == EDIT_TYPE_TRANSPOSITION:
                channel_dict[candidate_item] = spelling_corrector.channel_model(str1=edit[4][0], str2=edit[4][1],
                                                                                edit_type=EDIT_TYPE_TRANSPOSITION)
            if edit[0] == EDIT_TYPE_SUBSTITUTION:
                channel_dict[candidate_item] = spelling_corrector.channel_model(str1=edit[3], str2=edit[4],
                                                                                edit_type=EDIT_TYPE_SUBSTITUTION)

        for item in channel_dict:
            channel = channel_dict[item]
            if len(sentence) - 1 != word_i:  # not the end of this sentence
                bigram = math.pow(math.e,
                                  spelling_corrector.sentence_probability(
                                      sentence=sentence[word_i - 1] + item + sentence[word_i + 1],
                                      ngram_type=2))
            else:  # the end of this sentence
                bigram = math.pow(math.e,
                                  spelling_corrector.sentence_probability(sentence=sentence[word_i - 1],
                                                                          ngram_type=2))

            prob_dict[item] = channel * bigram * math.pow(10, 9)
        prob_dict = sorted(prob_dict, key=prob_dict.get, reverse=True)
        if not prob_dict:  # if P == []
            prob_dict.append(word)
        result_sentence = result_sentence + str(prob_dict[0]) + word_suffix + " "

    # After the model calculates the corrected sentence, the evaluation procedure needs to match the results
    ans_sentence = ans_file.readline().split("\t")[1]

    ans_set = set(nltk.word_tokenize(ans_sentence))
    result_set = set(nltk.word_tokenize(result_sentence))
    print("[INFO] result set: " + str(result_set))
    print("[INFO] answer set: " + str(ans_set))
    if ans_set == result_set:
        return 1
    else:
        return 0


if __name__ == "__main__":
    """
    Spelling Correction main function
    - Input file：the source file that needs this correction, the file path is <test_data_path>
    - Output file：the corrected result file is stored in <result_file_path>
    """

    print("[INFO] Spelling Correction starts...")

    spelling_corrector = SpellingCorrector()

    print("[INFO] Reading the test data...")
    test_data_file = open(test_data_path, 'r')
    test_data_lines = test_data_file.readlines()
    test_data_file.close()  # don't forget it

    ans_file = open(ans_file_path, "r")

    match_count = 0
    sentence_id = 0
    for test_data_line in test_data_lines:  # for each line of the test data
        sentence_id = sentence_id + 1
        res = sentence_spelling_correction(test_data_line)
        match_count = match_count + res
        print("Accuracy is : %.4f%%" % (match_count * 100.00 / sentence_id))

    print("[INFO] Spelling Correction ends")
