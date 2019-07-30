import ast
import math
import nltk
from nltk.corpus import reuters
from collections import Counter

# file path
test_data_path = "testdata.txt"
result_file_path = "result.txt"
vocabulary_path = "vocab.txt"

# constants
EDIT_TYPE_INSERTION = 0
EDIT_TYPE_DELETION = 1
EDIT_TYPE_SUBSTITUTION = 2
EDIT_TYPE_TRANSPOSITION = 3


class SpellingCorrector:
    # member fields

    word_list = []  # splice all the words in the corpus

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

    def load_corpus(self, corpus_name):
        if corpus_name == "reuters":
            return self.load_reuters_corpus()

    def load_reuters_corpus(self):
        """
        Get Reuters Corpus from nltk package
        :returns a list of words to splicing all the files in the corpus
        """

        nltk.download('reuters')
        reuters_fileids = reuters.fileids()
        print("[INFO] The length of reuters corpus is " + str(len(reuters_fileids)))
        reuters_categories = reuters.categories()
        print("[INFO] The length of reuters categories is " + str(len(reuters_categories)))
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
        :return: vocabulary list
        """



    def edit_distance(self, str1, str2):
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
        It is for the real word error.
        :param word: source word used to generate a candidate set
        :return: the candidate set of this word
        """
        candidates = dict()
        for word_list_item in self.word_list:
            edit_distance = self.edit_distance(word, word_list_item)
            if edit_distance <= 1:
                candidates[word_list_item] = edit_distance
                return sorted(candidates, key=candidates.get, reverse=False)
        return candidates

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
        file = open(file_path, 'r')
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
        corpus = ' '.join(self.word_list)  # use spaces to join all the elements in the list
        string = str1 + str2
        if edit_type == EDIT_TYPE_INSERTION:
            if str1 == '@':
                return self.add_mat[string] / corpus.count(' ' + str2)
            else:
                return self.add_mat[string] / corpus.count(str1)
        if edit_type == EDIT_TYPE_DELETION:
            return self.del_mat[string] / corpus.count(string)
        if edit_type == EDIT_TYPE_SUBSTITUTION:
            return self.sub_mat[string] / corpus.count(string)
        if edit_type == EDIT_TYPE_TRANSPOSITION:
            return self.rev_mat[string] / corpus.count(string)

    def count_ngrams(self):
        self.unigram = self.count_unigram(self.word_list)
        self.bigram = self.count_bigram(self.word_list)
        self.trigram = self.count_trigram(self.word_list)
        self.four_gram = self.count_four_gram(self.word_list)
        self.five_gram = self.count_five_gram(self.word_list)

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

    for test_data_line in test_data_lines:  # for each line of the test data
        test_data_line = test_data_line.split("\t")  # split the test_data_line according to the metadata
        sentence_id = test_data_line[0]  # sentence id
        n_error = test_data_line[1]  # the error number in this sentence
        sentence = test_data_line[2].lower()  # convert the original sentence to lowercase
        sentence = sentence.split(" ")  # split words in one sentence
        # There may be an empty element in the sentence list, and the last element may contain a period and "\n"
        sentence = [word for word in sentence if word != '']  # remove empty string in the sentence list
        # remove the "\n"
        for i in range(len(sentence)):
            sentence[i] = sentence[i].strip("\n")
        corrected_sentence = ""  # The sentence after the correction

        for word_i, word in enumerate(sentence):  # for each word in the original sentence
            word_candidates = spelling_corrector.get_candidates(word)  # get a candidate set for the word
            if word in word_candidates:
                corrected_sentence = corrected_sentence + word + ' '
                continue

            NP = dict()  # todo ?
            P = dict()  # todo ?

            for candidate_item in word_candidates:
                edit = spelling_corrector.edit_type(candidate_item, word)
                if edit is None:
                    continue
                if edit[0] == EDIT_TYPE_INSERTION:
                    NP[candidate_item] = spelling_corrector.channel_model(str1=edit[3][0], str2=edit[3][1],
                                                                          edit_type=EDIT_TYPE_INSERTION)
                if edit[0] == EDIT_TYPE_DELETION:
                    NP[candidate_item] = spelling_corrector.channel_model(str1=edit[4][0], str2=edit[4][1],
                                                                          edit_type=EDIT_TYPE_DELETION)
                if edit[0] == EDIT_TYPE_TRANSPOSITION:
                    NP[candidate_item] = spelling_corrector.channel_model(str1=edit[4][0], str2=edit[4][1],
                                                                          edit_type=EDIT_TYPE_TRANSPOSITION)
                if edit[0] == EDIT_TYPE_SUBSTITUTION:
                    NP[candidate_item] = spelling_corrector.channel_model(str1=edit[3], str2=edit[4],
                                                                          edit_type=EDIT_TYPE_SUBSTITUTION)

            for item in NP:
                channel = NP[item]
                if len(sentence) - 1 != word_i:  # not the end of this sentence
                    bigram = math.pow(math.e,
                                      spelling_corrector.sentence_probability(
                                          sentence=sentence[word_i - 1] + item + sentence[word_i + 1],
                                          ngram_type=2))
                else:
                    bigram = math.pow(math.e, spelling_corrector.sentence_probability(sentence[word_i - 1], 2))

                P[item] = channel * bigram * math.pow(10, 9)
            P = sorted(P, key=P.get, reverse=True)
            if not P:  # if P == []
                P.append("")
            corrected_sentence = corrected_sentence + str(P[0]) + " "
        print("[INFO] original sentence: " + str(sentence))
        print("[INFO] corrected sentence: " + corrected_sentence)
        # After the model calculates the corrected sentence, the evaluation procedure needs to match the results

    print("Spelling Correction ends")
