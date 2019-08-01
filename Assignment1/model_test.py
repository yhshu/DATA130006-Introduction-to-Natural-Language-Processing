vocabulary_path = "vocab.txt"


def load_vocabulary_test():
    vocab_file = open(vocabulary_path, "r")
    vocab_list = vocab_file.read().split("\n")
    vocab_file.close()
    print("vocab type: " + str(type(vocab_list)))
    print(vocab_list)


if __name__ == "__main__":
    load_vocabulary_test()
