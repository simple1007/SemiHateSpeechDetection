from utils.utils import cos_sim_normal as cos_sim
from gensim.models import FastText, Word2Vec
import numpy as np
import sys
import os
model = Word2Vec.load('ilbe_word2vec/ilbe_word2vec_nv.model')
# f = open('dictionary_.txt','w',encoding='utf-8')

folder = sys.argv[1]
enc = 'utf-8'
count = 0
point = 0.0
dup = {}
with open('dictionary/{}/1st.txt.txt'.format(folder),encoding=enc) as f:
    output = 'dictionary/{}/extend'.format(folder)
    if not os.path.exists(output):
        os.makedirs(output)
    
    output_f = open(output+'/dictionary_{}.txt'.format(folder),'w',encoding=enc)
    for l in f:
        if l.startswith('#'):
            continue
        l = l.strip()
        l = l.split('\t')
        point += float(l[1])
        count += 1
    
    point /= count

    f.seek(0)
    for l in f:
        if l.startswith('#'):
            l = l.split('\t')
            output_f.write(l[0][1:]+'\n')
            continue
        l = l.split('\t')
        words = model.wv.most_similar_cosmul(l[0])

        for w in words:
            if (w[1]*0.85) > point:
                if w[0] not in dup:
                    dup[w[0]] = 1
                    output_f.write(w[0]+'\n')

    output_f.close()