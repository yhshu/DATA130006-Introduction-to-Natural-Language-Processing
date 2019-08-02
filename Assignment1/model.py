import ast
import math
from collections import Counter
import nltk
from pyxdameraulevenshtein import damerau_levenshtein_distance

from Assignment1.corpus import load_reuters_corpus, load_gutenberg_corpus, load_brown_corpus
from util import rchop, lchop, legal_number

# THE CONSTANTS
suffix_list = ["'", "'d", "'flight", "'ll", "'re", "'s", "'ve"]
ngram = 2  # integer value, 1: unigram, 2: bigram, 3: trigram

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
        """
        initialize the spelling corrector
        1. load the corpus provided by nltk package
        2. count the n-gram
        3. load the confusion matrix from files, it's for the noisy channel model
        4. load the vocabulary
        """
        self.word_list.extend(self.load_corpus("reuters"))
        self.corpus = " ".join(self.word_list)  # use spaces to join all the elements in the list
        # load the corpus to create the word list
        # note that the network is needed to download the corpus

        self.count_ngrams()  # count the n-grams
        self.load_confusion_matrix()  # read the confusion matrix from files
        self.load_vocabulary()  # read the vocabulary from a file

    def load_corpus(self, corpus_name):
        """
        load the corpus
        :param corpus_name: the name of corpus provideed by nltk package
        :return: the corpus information
        """
        if corpus_name == "reuters":
            return load_reuters_corpus()
        if corpus_name == "gutenberg":
            return load_gutenberg_corpus()
        if corpus_name == "brown":
            return load_brown_corpus()

    def load_vocabulary(self):
        """
        load the vocabulary from file
        """
        vocab_file = open(vocabulary_path, "r")
        self.vocab_list = vocab_file.read().split("\n")
        vocab_file.close()
        print("[INFO] Reading vocabulary...")
        print(self.vocab_list[0:15])

    def get_candidates(self, word):
        """
        Damerau-Levenshtein edit distance is used to generate a candidate set of this word.
        :param word: source word used to generate a candidate set
        :return: the candidate set of this word
        """
        candidates = dict()
        for word_list_item in self.vocab_list:
            edit_distance = damerau_levenshtein_distance(word, word_list_item)
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
        :return: Noisy channel probability
        """
        string = str1.lower() + str2.lower()
        str1_count = self.corpus.count(str1)
        string_count = self.corpus.count(string)

        try:
            if edit_type == EDIT_TYPE_INSERTION:
                if str1 == "@":
                    if self.corpus.count(" " + str2) == 0:
                        return 0
                    return self.add_mat[string] / self.corpus.count(" " + str2)
                else:
                    if str1_count == 0:
                        return 0
                    return self.add_mat[string] / str1_count
            if edit_type == EDIT_TYPE_DELETION:
                if string_count == 0:
                    return 0
                return self.del_mat[string] / string_count
            if edit_type == EDIT_TYPE_SUBSTITUTION:
                if str1_count == 0:
                    return 0
                return self.sub_mat[string] / str1_count
            if edit_type == EDIT_TYPE_TRANSPOSITION:
                if string_count == 0:
                    return 0
                return self.rev_mat[string] / string_count
        except KeyError:
            return 0
        except ZeroDivisionError:
            return 0

    def count_ngrams(self):
        """
        Calculate the number of occurrences of a phrase in the corpus for n-gram models
        :return: several dict for n-gram models
        """
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

    def n_gram_MLE(self, given_words, predicted_words, gram_type=1):
        """
        Calculate the Maximum Likelihood Probability of n-grams with add-one smoothing

        Reference:
        - http://lintool.github.io/UMD-courses/CMSC723-2009-Fall/session9-slides.pdf
        """
        try:
            if gram_type == 1:
                return math.log((self.unigram[given_words] + 1) / (len(self.word_list) + len(self.unigram)))
            elif gram_type == 2:
                return math.log((self.bigram[predicted_words] + 1) / (self.unigram[given_words] + len(self.unigram)))
            elif gram_type == 3:
                return math.log(self.trigram[predicted_words] + 1) / (self.bigram[given_words] + len(self.unigram))
            elif gram_type == 4:
                return math.log(self.four_gram[predicted_words] + 1) / (self.trigram[given_words] + len(self.unigram))
            elif gram_type == 5:
                return math.log(self.five_gram[predicted_words] + 1) / (self.four_gram[given_words] + len(self.unigram))
        except ZeroDivisionError:
            return 0
        except KeyError:
            return 0

    def sentence_probability(self, sentence, ngram_type=1):
        """
        The ngram model is for phrases rather than entire sentences.
        For a sentence, we need to calculate the sum of the probabilities on the language model in all words.
        :param sentence: sentence requiring a probability
        :param ngram_type: integer 1: unigram; 2: bigram; 3: trigram
        :return: The sentence probability calculated based on the maximum likelihood estimation
                 of all words in the sentence in the LM.
        """
        sentence_word_list = sentence.lower().split()
        prob = 0  # sentence probability
        if ngram_type == 1:
            for i in range(len(sentence_word_list)):
                prob = prob + self.n_gram_MLE(sentence_word_list[i], gram_type=1)

        if ngram_type == 2:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 1:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i],
                                              sentence_word_list[i] + ' ' + sentence_word_list[i + 1], gram_type=2)

        if ngram_type == 3:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 2:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i] + ' ' + sentence_word_list[i + 1],
                                              sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2], gram_type=3)

        if ngram_type == 4:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 3:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2],
                                              sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2] + sentence_word_list[i + 3], gram_type=4)

        if ngram_type == 5:
            for i in range(len(sentence_word_list)):
                if i >= len(sentence_word_list) - 4:
                    break
                prob = prob + self.n_gram_MLE(sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2] + sentence_word_list[i + 3],
                                              sentence_word_list[i] + ' ' + sentence_word_list[i + 1] + ' ' +
                                              sentence_word_list[i + 2] + sentence_word_list[i + 3] +
                                              sentence_word_list[i + 4], gram_type=5)

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
        """
        Remove the word suffix and find the original form of the word
        :param word: word to be cleaned
        :return: a tuple, cleaned word and its suffix
        """
        word_ori = word
        if word not in self.vocab_list:  # if the word is not in the vocabulary
            word = word.strip(",.!?")  # delete punctuation, such as periods, commas
        for i in range(len(suffix_list)):
            (match, string) = rchop(word, suffix_list[i])
            if match:
                (_, suffix) = lchop(word_ori, string)
                return string, suffix

        (_, suffix) = lchop(word_ori, word)
        return word, suffix


def word_spelling_correction(word_i, word, non_word_index_list, n_non_word_error, n_error, result_sentence, sentence):
    if word_i not in non_word_index_list and n_non_word_error >= n_error:
        # it's a correct word
        result_sentence[word_i] = word
        return

    # it's non-word error or real word error
    word_cleaned, word_suffix = spelling_corrector.word_clean(word)
    word_candidates = spelling_corrector.get_candidates(word_cleaned)
    # get a candidate set for the word
    if word_cleaned in word_candidates or legal_number(word_cleaned):
        result_sentence[word_i] = word
        return

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

    for channel_word in channel_dict:
        channel = channel_dict[channel_word]
        if len(sentence) - 1 != word_i:  # not the end of this sentence
            sentence_prob = spelling_corrector.sentence_probability(
                sentence=sentence[word_i - 1] + " " + channel_word + " " + sentence[word_i + 1],
                ngram_type=ngram)
        else:  # the end of this sentence
            sentence_prob = spelling_corrector.sentence_probability(
                sentence=sentence[word_i - 1] + " " + channel_word,
                ngram_type=ngram)

        bigram = math.pow(math.e, sentence_prob)
        prob_dict[channel_word] = channel * bigram * math.pow(10, 9)
    prob_dict = sorted(prob_dict, key=prob_dict.get, reverse=True)
    if not prob_dict:  # if P == []
        prob_dict.append(word_cleaned)
    result_sentence[word_i] = str(prob_dict[0]) + word_suffix


def sentence_spelling_correction(test_data_line):
    """
    Generate a corrected sentence for a sentence in the test data.
    :param test_data_line: a sentence to be corrected
    :return: Returns 1 if it matches the answer, otherwise returns 0
    """
    test_data_line = test_data_line.split("\t")  # split the test_data_line according to the metadata
    sentence_id = test_data_line[0]  # sentence id
    n_error = int(test_data_line[1])  # the error number in this sentence
    sentence = test_data_line[2]  # the test sentence
    sentence = sentence.split(" ")  # split words in one sentence
    sentence = spelling_corrector.word_list_clean(sentence)

    result_sentence = [""] * len(sentence)  # The sentence after the correction

    # First, we determine the number of non-word errors by comparing the vocabulary.
    # If the number of errors is still greater than 0 except non-word errors, we begin to correct real word errors
    n_non_word_error = 0  # count the number of non-word errors
    non_word_index_list = []
    for word_i, word in enumerate(sentence):
        if word not in spelling_corrector.vocab_list and \
                spelling_corrector.word_clean(word)[0] not in spelling_corrector.vocab_list:
            # it's a non-word error
            n_non_word_error = n_non_word_error + 1
            non_word_index_list.append(word_i)

    print("[INFO] sentence id: " + str(sentence_id) + ", n_error: " + str(n_error) +
          ", non-word error: " + str(n_non_word_error))
    print("[INFO] original sentence: " + str(sentence))

    # for each word in the original sentence
    for word_i, word in enumerate(sentence):
        if word_i in non_word_index_list:
            word_spelling_correction(word_i, word, non_word_index_list,
                                     n_non_word_error, n_error, result_sentence, sentence)

    for word_i, word in enumerate(sentence):
        if word_i not in non_word_index_list:
            word_spelling_correction(word_i, word, non_word_index_list,
                                     n_non_word_error, n_error, result_sentence, sentence)

    result_sentence = " ".join(result_sentence)
    # After the model calculates the corrected sentence, the evaluation procedure needs to match the results
    ans_sentence = ans_file.readline().split("\t")[1]

    ans_set = set(nltk.word_tokenize(ans_sentence))
    result_set = set(nltk.word_tokenize(result_sentence))
    print("[INFO] result set: " + str(result_set))
    print("[INFO] answer set: " + str(ans_set))

    # write the result sentence to a file
    with open(result_file_path, 'a') as result_file:
        print(str(sentence_id) + "\t" + result_sentence, file=result_file)

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
    test_data_file.close()  # don't forget to close it

    ans_file = open(ans_file_path, "r")

    match_count = 0
    sentence_id = 0
    for test_data_line in test_data_lines:  # for each line of the test data
        sentence_id = sentence_id + 1
        res = sentence_spelling_correction(test_data_line)
        match_count = match_count + res
        print("Accuracy is: %.4f%%" % (match_count * 100.00 / sentence_id))

    print("[INFO] Spelling Correction ends")
