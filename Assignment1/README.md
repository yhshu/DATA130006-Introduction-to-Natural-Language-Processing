# 作业1 拼写纠正

## 1. 作业要求

- 需要使用第三章讲义中提到的信道模型和语言模型。
- 关于评估，详见 testdata.txt.

### 1.1 提供的文件

- testdata.txt
    - 这是由新闻文章中提取的 1000 个样本句子组成的文本文件。
    - 每行包含一个包含三个项目的样本，包括句子ID，错误单词数和句子。它们由制表符分隔。
    - 其中，50个实例包含真正的单词错误。
- ans.txt
    - 包含拼写纠正后 1000 个句子的答案文件。
    - 每行包含一个包含两个项目的样本，包括句子ID和纠正后的句子。
- vocab.txt
    - 用于非单词错误检测的词典。必须使用该文件。
- eval.py
    - 这是一个供您参考的包含评估程序的 Python 文件。答案和结果文件的路径写在代码中。请阅读它。
    - 运行"eval.py"，它将给出一个准确度数字。根据此指标改进您的程序。
    - 运行时，确保“ans.txt”和“result.txt”与“eval.py”位于同一目录中。
    - 可能包含一些错误。如果找到，请发邮件联系。
如果您找到，请给我们发电子邮件！
- 语言模型（LM）
    - SRILM 的编译文件。

### 1.2 提交
- 生成一个zip文件并将其命名为“sid_homework-1.zip”。
- 应包括名为 program 的目录，输出文件“result.txt”和书面报告“spell correction.pdf”。
- 程序：代码应该用 Python 编写。
- 输出文件：每行包括一个由两个项组成的实例，即句子id和校正后的句子。用制表符分隔它们。
- 报告：应当使用英文编写且在4页以内。

### 1.3 作业评分
- 根据以下准则进行评分：
    - 最终准确度 20%
    - 程序 30%
    - 报告 40%
    - LM 实现 10%
- 如果使用工具集来实现语言模型，最高分不会超过满分的 90%；如果自己编写语言模型，有可能获得满分。

## 2. 参考资料

- [How to Write a Spelling Corrector by Norvig](
https://norvig.com/spell-correct.html)
- https://github.com/jbhoosreddy/spellcorrect

## 3. 混淆矩阵 Confusion Matrix

程序使用的混淆矩阵来自于论文《A Spelling Correction Program Based on a Noisy Channel Model》.

混淆矩阵的数据存储在 `confusion_matrix` 目录下。
- del[X, Y] = Deletion of Y after X
- add[X, Y] = Insertion of Y after X
- sub[X, Y] = Substitution of X (incorrect) for Y (correct)
- rev[X, Y] = Reversal of XY

## 4. ngram 语言模型

使用拉普拉斯平滑的 ngram 极大似然估计，参考：
- http://lintool.github.io/UMD-courses/CMSC723-2009-Fall/session9-slides.pdf

## 5. 编辑距离

编写在仓库根目录的 util.py 文件中。参考：
- [Levenshtein 距离](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [python-Levenshtein documentation](https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html)
- [pyxDamerauLevenshtein](https://pypi.org/project/pyxDamerauLevenshtein/)
- [Damerau-Levenshtein 距离](https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance)