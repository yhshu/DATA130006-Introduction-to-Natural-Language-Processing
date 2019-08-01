# coding=utf-8
import nltk

ans_path = './ans.txt'  # 真实结果文件路径
result_path = './result.txt'  # 模型预测结果
download_needed = False  # default: True

if __name__ == "__main__":
    if download_needed:
        nltk.download('punkt')
    ans_file = open(ans_path, 'r')  # 真实结果文件
    result_file = open(result_path, 'r')  # 模型预测结果文件
    match_count = 0
    n_sentence = 1000  # 句子数量
    for i in range(n_sentence):
        ans_line = ans_file.readline().split('\t')[1]
        ans_set = set(nltk.word_tokenize(ans_line))
        result_line = result_file.readline().split('\t')[1]
        result_set = set(nltk.word_tokenize(result_line))
        if ans_set == result_set:
            match_count += 1
    print("Accuracy is : %.4f%%" % (match_count * 100.00 / n_sentence))
