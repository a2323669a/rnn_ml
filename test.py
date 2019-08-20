import jieba
import io
import jieba.analyse

with io.open("./data/sp.txt", encoding='gbk') as f:
    text = f.read()
a = jieba.analyse.extract_tags(text, allowPOS=('nr2'))
