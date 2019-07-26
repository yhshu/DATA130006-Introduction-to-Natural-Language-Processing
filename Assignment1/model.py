
from nltk.corpus import reuters

# constants
test_data_path = 'testdata.txt'
result_file_path = 'result.txt'


def load_reuters_corpus():
    """
    Get Reuters Corpus
    """
    reuters_fileids = reuters.fileids()
    print("The length of reuters corpus is " + str(len(reuters_fileids)))


def get_candidates(word):
    """
    Damerau-Levenshtein edit distance is used to generate a candidate set of this word.
    It is for the real word error.
    :param word: source word used to generate a candidate set
    :return: the candidate set of this word
    """
    candidates = dict()
    return candidates


if __name__ == "__main__":
    """
    Spelling Correction main function
    - Input file：the source file that needs this correction, the file path is <test_data_path>
    - Output file：the corrected result file is stored in <result_file_path>
    """

    print("Spelling Correction starts...")
    test_data_file = open(test_data_path, 'r')
    test_data_lines = test_data_file.readlines()
    for test_data_line in test_data_lines:  # for each line of the test data
        line_sentence_id = test_data_line[0]  # sentence id
        line_n_error = test_data_line[1]  # the error number in this sentence
        line_test_sentence = test_data_line[2].lower()  # Convert the original sentence to lowercase
        test_data_line_correct = ""  # The sentence after the correction
        for word_index, word in enumerate(line_test_sentence):  # for each word in the original sentence
            word_candidates = get_candidates(word)  # 获得该单词的候选词

    print("Spelling Correction ends")
