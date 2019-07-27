import nltk
from nltk.corpus import reuters

# file path
test_data_path = 'testdata.txt'
result_file_path = 'result.txt'

# constants
EDIT_TYPE_INSERTION = 0
EDIT_TYPE_DELETION = 1
EDIT_TYPE_SUBSTITUTION = 2
EDIT_TYPE_TRANSPOSITION = 3


class SpellingCorrector:
    word_list = []  # splice all the words in the corpus

    def __init__(self):
        self.word_list = self.load_corpus("reuters")

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

    def edit_distance(self, str1, str2):
        """
        Calculate Damerau-Levenshtein Edit Distance for two string

        Reference:
        https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance

        :param str1: string 1
        :param str2: string 2
        :return: the Damerau-Levenshtein Edit Distance between str1 and str2
        """
        str1 = '#' + str1
        str2 = '#' + str2
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
                        replacer = '#'
                        replaced = '#' + error
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
                        replacer = '#'
                        replaced = '#' + error
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

    def channel_model(self, str1, str2, edit_type):
        corpus = ' '.join(self.word_list)  # use spaces to join all the elements in the list
        if edit == EDIT_TYPE_INSERTION:




if __name__ == "__main__":
    """
    Spelling Correction main function
    - Input file：the source file that needs this correction, the file path is <test_data_path>
    - Output file：the corrected result file is stored in <result_file_path>
    """

    print("Spelling Correction starts...")

    spelling_corrector = SpellingCorrector()

    test_data_file = open(test_data_path, 'r')
    test_data_lines = test_data_file.readlines()
    for test_data_line in test_data_lines:  # for each line of the test data
        line_sentence_id = test_data_line[0]  # sentence id
        line_n_error = test_data_line[1]  # the error number in this sentence
        line_test_sentence = test_data_line[2].lower()  # convert the original sentence to lowercase
        test_data_line_correct = ""  # The sentence after the correction

        for word_index, word in enumerate(line_test_sentence):  # for each word in the original sentence
            word_candidates = spelling_corrector.get_candidates(word)  # get a candidate set for the word
            if word in word_candidates:
                test_data_line_correct = test_data_line_correct + word + ' '
                continue

            for candidate_item in word_candidates:
                edit = spelling_corrector.edit_type(candidate_item, word)
                if edit is None:
                    continue
                # if edit[0] == "Insertion":
                #     NP[item] = sc.channelModel(edit[3][0], edit[3][1], 'add')
                # if edit[0] == 'Deletion':
                #     NP[item] = sc.channelModel(edit[4][0], edit[4][1], 'del')
                # if edit[0] == 'Reversal':
                #     NP[item] = sc.channelModel(edit[4][0], edit[4][1], 'rev')
                # if edit[0] == 'Substitution':
                #     NP[item] = sc.channelModel(edit[3], edit[4], 'sub')

    print("Spelling Correction ends")
