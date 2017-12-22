
> 少年壮志不言愁
> 劝君惜取少年时

#### [原文链接](https://www.jianshu.com/p/e754d10f4fe6)


![图片来自网络](http://upload-images.jianshu.io/upload_images/3818161-f545b252c2d70e45.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**贝叶斯定理：**
贝叶斯定理是关于随机事件A和B的条件概率（或边缘概率）的一则定理。其中P(A|B)是在B发生的情况下A发生的可能性。关于贝叶斯理论的详细推理，可以参考这篇[文章](http://www.jianshu.com/p/c59851b1c0f3)。
> P(A丨B)=P(A)P(B丨A)/P(B)

## 小试牛刀
这里选择当当网书评价（好评、差评）应用贝叶斯分类算法，其中差评数据10w条，好评数据11w条，数据保存到trainset.csv[数据下载链接](https://github.com/xhades/rates_classify/tree/master/rates_classify/data)
![训练集trainset.csv.png](http://upload-images.jianshu.io/upload_images/3818161-45092258cd9c6278.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
训练集中包括差评和好评数据共221968，其中包括无效数据及空行，后面将被清除
训练集第一行`header`包括两个字段`rate`即评论正文和评论类型`type`即差评与好评

#### 1. 首先对抓取的数据清洗，删除`空格`、`\u3000`、 `\xa0`等字符
  ```python
    def cleanTrianSet(filepath):
        """
        清洗句子中空行、空格
        目前采用将所有数据读取到内存，后续思考其他高效方式
        """
        # 删除评论上面的 \n
        fileDf = pd.read_csv(filepath, keep_default_na=False)
        fileDf["rate"] = fileDf["rate"].apply(lambda x: x.replace("\n", ""))
        linelist = fileDf.values.tolist()
        filelines = [ _[0] + "," + _[-1] for _ in linelist]
        cleaned_lines = map(lambda x: x.translate({ord('\u3000'): '', ord('\r'): '', ord('\xa0'): None,
                                                    ord(' '): None}), filelines[1:])  # 更加优雅的方式 在这个问题中是比较快的方式
        return cleaned_lines  # 返回一个map对象
  ```

#### 2. 使用开源分词工具**jieba分词**对正负面语料进行分词，分词过程中删除了空行等。分词代码`tools/jieba_split.py`，分词结果如下图
![分词后数据集.png](http://upload-images.jianshu.io/upload_images/3818161-aa9e6b08f7d2cd7e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
同时将label写入`data/label.txt`
![label.txt.png](http://upload-images.jianshu.io/upload_images/3818161-8437c852d7c70deb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 3.使用`Word2Vec`对分词数据集训练词向量
参数设置说明
- size=128： 设置词向量维度为128，是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百
- window=12：训练窗口设置为12，即考虑一个词前五个词和后五个词的影响
- min_count=10：词频小于该值的词就会被舍弃
- sg：设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
```python
#!/usr/bin/env python
# -*-coding:utf-8-*-
"""
@Time: 17-11-20 
@author: xhades
@version: v0.1
"""
from gensim.models import word2vec

sentence = word2vec.Text8Corpus('../data/splited_words.txt')
model = word2vec.Word2Vec(sentence, size=128, min_count=10, sg=1, window=12, workers=8)
model.wv.save_word2vec_format("../data/embedding.txt", binary=False, )
model.save("../Model/word2vec.model")
```
形成`embedding.txt`词嵌入文件，即保存了所有词的词向量

#### 4.数据预处理        
代码模块`preprocessing.py`
- 代码解析1
  ```python
  embeddingMtx = np.zeros((212841, 128), dtype='float32')
  ```
    这里构造一个词嵌入矩阵用于存放每条评论的句子矩阵（句子矩阵由词向量表示），其中212841是数据集评论数量，128是词向量维度

- 代码解析2
  ```python
   wordsEmbed = map(lambda word: embedding_lookup(word, embDict), words)
  ```
    `embedding_lookup()`方法会在词向量中寻找对应词的向量，如果某个词没有在词向量文件中就在[-0.5, 0.5]之间随机生成128维的矩阵
  ```python
  def embedding_lookup(voc, embDict):
      embedding = embDict.get(voc, [random.uniform(-0.5, 0.5) for i in range(128)])
      return embedding
  ```
- 代码解析3
    最后通过`embeddingMtx[count] = wordEmbeddingMtx[0]`将每一行数据放入词嵌入矩阵`embeddingMtx`中
- 完整代码如下
  ```python
  import codecs
  import numpy as np
  import pickle
  from tools.utils import embedding_lookup

  np.set_printoptions(threshold=np.inf)


  # 将训练文本数据转换成embedding词矩阵
  def build_embedding():
      # 词向量形式转变成字典
      with open("data/embedding.txt") as embFile:
          embLines = embFile.readlines()
      embDict = {_.strip("\n").split(" ")[0]: _.strip("\n").split(" ")[1:] for _ in embLines[1:]}

      # 加载splited  word文件
      fileData = codecs.open("data/splited_words.txt", "r", encoding="utf-8")

      # embedding文件
      embeddingMtx = np.zeros((212841, 128), dtype='float32')
      count = 0
      fileLine = fileData.readline()

      while fileLine:
          fileLine = fileLine.strip()
          if fileLine :
              words = fileLine.split(" ")
              # 对应词向量列表
              wordsEmbed = map(lambda word: embedding_lookup(word, embDict), words)
              # 列表转成矩阵, 序列化写入文件
              wordEmbeddingMtx = np.matrix(list(wordsEmbed))
              embeddingMtx[count] = wordEmbeddingMtx[0]

              fileLine = fileData.readline()
              count += 1
              continue

          fileLine = fileData.readline()
      fileData.close()
      print("End.....")
      # print(embeddingMtx)
      with open("Res/char_embedded.pkl", "wb") as file_w:
          pickle.dump(embeddingMtx, file_w)
  ```

5.训练数据
在`sklearn`中，提供了3中朴素贝叶斯分类算法：GaussianNB(高斯朴素贝叶斯)、MultinomialNB(多项式朴素贝叶斯)、BernoulliNB(伯努利朴素贝叶斯)

我这里主要选择使用伯努利模型的贝叶斯分类器来进行短评分类。

并且按照7：3的比例划分训练集和测试集
```python
import numpy as np
from numpy import array, argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import BernoulliNB
import pickle

np.set_printoptions(threshold=np.inf)


# 训练集测试集 3/7分割
def train(xFile, yFile):
    with open(xFile, "rb") as file_r:
        X = pickle.load(file_r)

    # 读取label数据，并且使用LabelEncoder对label进行编号
    with open(yFile, "r") as yFile_r:
        labelLines = [_.strip("\n") for _ in yFile_r.readlines()]
    values = array(labelLines)
    labelEncoder = LabelEncoder()
    integerEncoded = labelEncoder.fit_transform(values)
    integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)
    # print(integerEncoded)

    # 获得label 编码
    Y = integerEncoded.reshape(212841, )
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # 训练数据
    clf = BernoulliNB()
    clf.fit(X_train, Y_train)

    # 测试数据
    predict = clf.predict(X_test)
    count = 0
    for p, t in zip(predict, Y_test):
        if p == t:
            count += 1
    print("Accuracy is:", count/len(Y_test))
```

最终使用朴素贝叶斯分类器最终准确率在73%左右，分类效果还算不错=。=

完整代码查看[rates_classify](https://github.com/xhades/rates_classify)






