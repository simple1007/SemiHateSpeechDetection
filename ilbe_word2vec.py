from gensim.models.fasttext import Word2Vec,FastText
import re
data = []

with open('preprocessing/ht_n.txt',encoding='utf-8') as f:
    for l in f:
        l = l.strip()
        te = l
        l = re.sub(' +',' ',l)
        l = l.split(' ')
        if len(l) > 0 or te.strip() != '':
            data.append(l)

ftmodel = Word2Vec(min_count=5,vector_size=300,workers=6,window=7)
# ftmodel = FastText(min_count=1,vector_size=300,workers=6)
ftmodel.build_vocab(data)
ftmodel.train(data,total_examples=len(data),epochs=20)

ftmodel.save('ilbe_word2vec/ilbe_word2vec_nv.model')