import nltk
from nltk.corpus import gutenberg, reuters, brown

download_needed = False  # default: True, when the corpus is used for the first time, downloading is necessary.


def load_gutenberg_corpus():
    """
    Get Gutenberg Corpus from nltk package
    :return: a list of words to splicing all the files in the corpus
    """
    if download_needed:
        nltk.download("gutenberg")
    else:
        print(
            "[INFO] The local corpus is used, if you have not already downloaded the corpus, set the download_needed = True after connecting to the Internet")
    gutenberg_fileids = gutenberg.fileids()
    print("[INFO] The length of gutenberg corpus is " + str(len(gutenberg_fileids)))

    word_list = []
    for file_id in gutenberg_fileids:
        file_word = nltk.corpus.gutenberg.words(file_id)  # words in this file
        file_word_lowercase = [w.lower() for w in file_word]  # convert words to lowercase
        word_list.extend(file_word_lowercase)
    print("[INFO] Load Gutenberg corpus completed")
    return word_list


def load_reuters_corpus():
    """
    Get Reuters Corpus from nltk package
    :returns a list of words to splicing all the files in the corpus
    """

    if download_needed:
        nltk.download("reuters")
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


def load_brown_corpus():
    if download_needed:
        nltk.download("brown")
    else:
        print(
            "[INFO] The local corpus is used, if you have not already downloaded the corpus, set the download_needed = True after connecting to the Internet")
    brown_fileids = brown.fileids()
    print("[INFO] The length of brown corpus is " + str(len(brown_fileids)))
    brown_categories = brown.categories()
    print("[INFO] The length of brown categories is " + str(len(brown_categories)))
    print("[INFO] Checking if the corpus file has been obtained...")
    print(brown_fileids[0:5])  # check if the files has been downloaded

    word_list = []
    for file_id in brown_fileids:
        file_word = nltk.corpus.brown.words(file_id)  # words in this file
        file_word_lowercase = [w.lower() for w in file_word]  # convert words to lowercase
        word_list.extend(file_word_lowercase)
    print("[INFO] Load Brown corpus completed")
    return word_list
