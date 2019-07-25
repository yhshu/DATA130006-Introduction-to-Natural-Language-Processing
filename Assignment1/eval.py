# coding=utf-8
import nltk

anspath = './ans.txt'                # 真实结果
resultpath = './result.txt'          # 模型预测结果
ansfile = open(anspath, 'r')
resultfile = open(resultpath, 'r')
count = 0
n_sentence = 1000                    # 句子数量
for i in range(n_sentence):
    ansline = ansfile.readline().split('\t')[1]
    ansset = set(nltk.word_tokenize(ansline))
    resultline = resultfile.readline().split('\t')[1]
    resultset = set(nltk.word_tokenize(resultline))
    if ansset == resultset:
        count += 1
print("Accuracy is : %.2f%%" % (count * 1.00 / 10))
