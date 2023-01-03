import sys
sys.path.append('D:/SentenceSimilarityPredict/HeadTail_Tokenizer_POSTagger')

from utils.utils import cos_sim#cos_sim_per as cos_sim
# from konlpy.tag import Okt
from HeadTail_Tokenizer_POSTagger.head_tail import analysis

import tensorflow as tf
import numpy as np
import sentencepiece as spm

# o = Okt()
maxlen = 300

sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_ht_spm_model/ilbe_spm.model'
sp.load(vocab_file)

gender = np.load('emb/gender_emb.npy')
society = np.load('emb/soc_emb.npy')
age = np.load('emb/age_emb.npy')
toxic = np.load('emb/toxic_emb.npy')
total = np.load('emb/total_emb.npy')
# age = np.load('emb/age.npy')
# location = np.load('emb/location.npy')
# gender = np.load('emb/gen_sent.npy')
model = tf.keras.models.load_model('6_hate_speech_model')
model.load_weights('6_hate_speech_weights')
# model = tf.saved_model.load('1_hate_speech_model')
# model.summary()
# model.build((None,maxlen))
tag = ['N','V']
# from konlpy.tag import Okt
# o = Okt()
# ['▁나', '▁밥', '▁학교']
while True:
    x = input("input sentence: ")
    if x.lower() == 'exit':
        break
    x = analysis(x)
    print(x)
    x = x[0].split(' ')
    # print(x)
    # continue
    temp = []
    pos = []
    nv_chk = -1
    nv_dict = {}
    for nv_index,xx in enumerate(x):
        xx_ = xx.split('+')[0]
        xx_ = xx_.split('/')
        
        xx__ = ['TEMP']
        if '+' in xx:
            xx__ = xx.split('+')[1]
            xx__ = xx__.split('/')
        
        if xx_[1][0] in tag:
            # temp.append(xx_[0])
            if xx_[1][0] == 'V' and len(xx_[0]) > 1:# and (xx__[0][-1] == '다' or xx_[0][-1] == '했' or xx_[0][-1] == '하' or xx_[0][-1] == '한') and len(xx_[0]) >= 1: #and (xx_[0][-1] == '하' or xx_[0][-1] == '했' ):
                # if nv_chk > -1 and len(temp[-1]) == 1:
                #     temp[-1] = temp[-1] + '_' + xx_[0]
                #     pos[-1] = pos[-1] + '_V'
                #     # nv_dict[nv_chk] = nv_chk+1
                # else:# xx_[1][0] == 'V':
                #     if len(xx_[0]) == 1:
                #         continue
                #     temp.append(xx_[0])
                #     pos.append(xx_[1][0])
                # continue
                temp.append(xx_[0])
                pos.append(xx_[1][0])
                nv_chk = -1

            elif xx_[1][0] == 'N':
                if len(xx_[0]) == 1:
                    continue
                # if len(temp) !=0 and '_' not in temp[-1] and pos[-1] == 'N' and len(temp[-1]) == 1:
                #     # print('1',temp)
                #     # print('1',pos)
                #     temp.pop(-1)
                #     pos.pop(-1)
                #     # print('2',temp)
                #     # print('2',pos)
                nv_chk = nv_index
                temp.append(xx_[0])
                pos.append(xx_[1][0])
            else:
                nv_chk = -1
        # if '+' in xx:
        #     xx_ = xx.split('+')[1]
        #     xx_ = xx_.split('/')
        #     # if xx[1][0] in tag:
        #     temp.append(xx_[0])

    x = ' '.join(temp)#.replace('_','')
    nv_x = ' '.join(temp)
    print(x)
    # continue
    
    x = x.replace('_','')
    # temp = x.split(' ')1
    
    tt = sp.encode_as_pieces(x)
    x = sp.encode_as_ids(x)
    # print(tt)
    # print(tt)
    x = x + [0] * (maxlen - len(x))
    x = x[:maxlen]
    x = np.array([x])
    # print(np.array(x).shape)
    inputs = tf.keras.layers.Input(shape=(maxlen),name='inputs:0',dtype=tf.int64)
    # x = inputs(np.array([x]))
    gen,soc,toxic,age = model(inputs)
    # print(gen.shape)
    model = tf.keras.Model(inputs=inputs, outputs=[gen,soc,toxic,age])
    model.summary()
    gen,soc,toxic,age = model(x)
    print("gen: ",gen,"soc: ",soc,"toxic: ",toxic,"age: ",age)
    # print(pred)
    continue
    # exit()
    temp_piece = []
    # cnt = 0
    start = 0
    tmp = 0
    # start = 1
    # temp_piece.append([tmp])
    for index, t in enumerate(tt):
        if (t.startswith('▁') or t == '▁'):
            if index != 0:
                temp_piece[-1].append(start-1)
            temp_piece.append([tmp])
        # elif index != 0:
        #     temp_piece[-1].append(tmp)    
        if index == len(tt)-1:
            temp_piece[-1].append(start)
        tmp = start + 1
        start += 1
        # cnt += 1
    print(temp_piece)
    # for pr in pred[0]:
    v_t = []
    print(pred.shape)
    for tp in temp_piece:
        sm = np.sum(pred[0][tp[0]:tp[1] + 1],axis=0)
        for i in range(tp[0],tp[1]+1):
            print(pred[0][i:i+1].shape,society.shape)
            print(tt[i],cos_sim(age,pred[0][i:i+1][0]))
        # sm = np.zeros(128)
        # for i in range(tp[0],tp[1]+1):
        #     sm += pred[0][i]
        # print(sm)
        sm = sm / (tp[1] - tp[0] + 1)
        print(sm.shape)
        v_t.append(sm)
    # print(v_t,gender)
        # print(pr,"젠더 혐오:",cos_sim(pr,gender),'사회/정치 혐오:',cos_sim(pr,society))
    # print(temp)

    for tp,vt,p in zip(temp,v_t,pos):
        # print(vt.shape)
        # print(gender.shape)
        # def pearson_similarity(a, b):
        #     return np.dot((a - np.mean(a)), (b - np.mean(b))) / ((np.linalg.norm(a - np.mean(a))) * (np.linalg.norm(b - np.mean(b))))

        # hp = 0.0
        
        gd = cos_sim(vt,gender)
        sc = cos_sim(vt,society)
        te = cos_sim(vt,toxic)
        ae = cos_sim(vt,age)
        tt = cos_sim(vt,total)
        # print(gd,sc,ae)
        if False:
            if p == 'N':
                gd = (gd - (-1.5)) / (1 - (-1.5))
                sc = (sc - (-1.5)) / (1 - (-1.5))
                ae = (ae - (-1.5)) / (1 - (-1.5))
            elif p == 'V' or p=='N_V':
                gd = (gd - (-0.5)) / (1 - (-0.5))
                sc = (sc - (-0.5)) / (1 - (-0.5))
                ae = (ae - (-0.5)) / (1 - (-0.5))
        # if gd > 1.0:
        #     gd = 1.0
        # if sc > 1.0:
        #     sc = 1.0

        # gd = pearson_similarity(vt,gender)
        # sc = pearson_similarity(vt,society)

        # gd = gd / (gd + sc)
        # sc = sc / (gd + sc)
        if True:#(p == 'N' and (gd > 0.7 or sc > 0.7 or ae > 0.7)) or ((p == 'V' or p == 'N_V') and (gd > 0.77 or sc > 0.77 or ae > 0.77)):
            print(tp,"젠더 혐오:",gd,'사회/정치 혐오:',sc,'욕설:',te,'연령:',ae,'전체',tt)
        # print(tp,"젠더 혐오:",cos_sim(vt,gender),'사회/정치 혐오:',cos_sim(vt,society))
        # temp.append(xx[0])
    # break
    # x = ' '.join(temp)
    # # x = ' '.join(x)
    # print(x)
    # x = sp.encode_as_ids(x)
    # x = x + [0] * (maxlen - len(x))
    # x = x[:maxlen]

    # pred = model(np.array([x]))
    # print("젠더 혐오:",cos_sim(pred[0],gender),'사회/정치 혐오:',cos_sim(pred[0],society))
    # # print("젠더 혐오:",cos_sim(pred[0],gender),"정치 혐오:",cos_sim(pred[0],society),"연령 혐오:",cos_sim(pred[0],age),"지역 혐오:",cos_sim(pred[0],location))