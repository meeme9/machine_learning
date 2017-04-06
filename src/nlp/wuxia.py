# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import scipy as sp
import codecs
import gensim
import jieba

import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

data_dir = '/Users/xiaotz/src/machine_learning/data/wuxia'
font_yahei_consolas = FontProperties(fname=os.path.join(data_dir,"SimHei.ttf"))


def get_novel_names():
    with codecs.open(os.path.join(data_dir, 'list.txt'), encoding="utf8") as inf:
        data = [line.strip() for line in inf]
    return filter(None, data)


def get_menpai_names():
    with codecs.open(os.path.join(data_dir, 'menpai.txt'), encoding="utf8") as inf:
        data = [line.strip() for line in inf]
    return filter(None, data)


def get_kungfu_names():
    with codecs.open(os.path.join(data_dir, 'kungfu.txt'), encoding="utf8") as inf:
        data = [line.strip() for line in inf]
    return filter(None, data)


def get_names():
    with codecs.open(os.path.join(data_dir, 'name.txt'), encoding="utf8") as f:
        data = [line.strip() for line in f]
    novels = [x.split(u' ')[0] for x in data[::3]]
    names = data[1::3]
    novel_names = {k: v.split(' ') for k, v in zip(novels, names)}
    return novel_names


def find_main_charecters(novel, n, novel_names, num=10):
    with codecs.open(os.path.join(data_dir,'novel/{}.txt'.format(novel)), encoding="utf8") as f:
        data = f.read()
    chars = novel_names[n]
    count = np.array(map(lambda x: data.count(x), chars))
    idx = np.argsort(count)
    plt.barh(range(num), np.array(count)[idx[-num:]].tolist(), color='red', align='center')
    plt.title(n, fontsize=14, fontproperties=font_yahei_consolas)
    plt.yticks(range(num), np.array(chars)[idx[-num:]], fontsize=14, fontproperties=font_yahei_consolas)
    plt.show()

novel_names = get_names()
#find_main_charecters('天龙八部', u"天龙八部", novel_names)
#find_main_charecters("射雕英雄传", u"射雕英雄传", novel_names)
#find_main_charecters("神雕侠侣", u"神雕侠侣", novel_names)
#find_main_charecters("倚天屠龙记", u'倚天屠龙记', novel_names)


for x in get_novel_names():
    jieba.add_word(x)

for x in get_kungfu_names():
    jieba.add_word(x)

for x in get_menpai_names():
    jieba.add_word(x)

for _, names in novel_names.iteritems():
    for name in names:
        jieba.add_word(name)

sentences = []
for novel in get_novel_names():
    print novel
    print type(novel)
    print u"处理：{}".format(novel)
    with codecs.open(data_dir.decode('UTF-8') + u'/novel/{}.txt'.format(novel), encoding="utf8") as f:
        sentences += [list(jieba.cut(line.strip())) for line in f]


print 'start train'
model = gensim.models.Word2Vec(sentences,
                               size=50,
                               window=5,
                               min_count=5,
                               workers=4)
print 'end train'

for k, s in model.most_similar(positive=[u"乔峰", u"萧峰"]):
    print k, s

for k, s in model.most_similar(positive=[u"阿朱"]):
    print k, s

for k, s in model.most_similar(positive=[u"丐帮"]):
    print k, s

for k, s in model.most_similar(positive=[u"降龙十八掌"]):
    print k, s

def find_relationship(a, b, c):
    """
    返回 d
    a与b的关系，跟c与d的关系一样
    """
    d, _ = model.most_similar(positive=[c, b], negative=[a])[0]
    print u"给定“{}”与“{}”，“{}”和“{}”有类似的关系".format(a, b, c, d)

find_relationship(u"段誉", u"段公子", u"乔峰")

# 情侣对
find_relationship(u"郭靖", u"黄蓉", u"杨过")
# 岳父女婿
find_relationship(u"令狐冲", u"任我行", u"郭靖")
# 非情侣
find_relationship(u"郭靖", u"华筝", u"杨过")

# 韦小宝
find_relationship(u"杨过", u"小龙女", u"韦小宝")
find_relationship(u"令狐冲", u"盈盈", u"韦小宝")
find_relationship(u"张无忌", u"赵敏", u"韦小宝")

find_relationship(u"郭靖", u"降龙十八掌", u"黄蓉")
find_relationship(u"武当", u"张三丰", u"少林")
find_relationship(u"任我行", u"魔教", u"令狐冲")


all_names = np.array(filter(lambda c: c in model, novel_names[u"天龙八部"]))
word_vectors = np.array(map(lambda c: model[c], all_names))
from sklearn.cluster import KMeans
N = 3

label = KMeans(N).fit(word_vectors).labels_

for c in range(N):
    print u"\n类别{}：".format(c+1)
    for idx, name in enumerate(all_names[label==c]):
        print name,
        if idx % 10 == 9:
            print
    print


N = 4

c = sp.stats.mode(label).mode

remain_names = all_names[label!=c]
remain_vectors = word_vectors[label!=c]
remain_label = KMeans(N).fit(remain_vectors).labels_

for c in range(N):
    print u"\n类别{}：".format(c+1)
    for idx, name in enumerate(remain_names[remain_label==c]):
        print name,
        if idx % 10 == 9:
            print
    print

import scipy.cluster.hierarchy as sch



all_names = np.array(filter(lambda c: c in model, novel_names[u"倚天屠龙记"]))
word_vectors = np.array(map(lambda c: model[c], all_names))
Y = sch.linkage(word_vectors, method="ward")
_, ax = plt.subplots(figsize=(100, 400))
Z = sch.dendrogram(Y, orientation='right')
idx = Z['leaves']
ax.set_xticks([])
ax.set_yticklabels(all_names[idx],
                   fontproperties=font_yahei_consolas)
ax.set_frame_on(False)
plt.show()


all_names = np.array(filter(lambda c: c in model, get_kungfu_names()))
word_vectors = np.array(map(lambda c: model[c], all_names))
Y = sch.linkage(word_vectors, method="ward")
_, ax = plt.subplots(figsize=(100, 350))
Z = sch.dendrogram(Y, orientation='right')
idx = Z['leaves']
ax.set_xticks([])
ax.set_yticklabels(all_names[idx], fontsize=8,
                   fontproperties=font_yahei_consolas)
ax.set_frame_on(False)
plt.show()

