from utils.utils import cos_sim_normal as cos_sim
from gensim.models import FastText, Word2Vec
import numpy as np
import sys
model = Word2Vec.load('ilbe_word2vec/ilbe_word2vec_nv.model')
# f = open('dictionary_.txt','w',encoding='utf-8')

if len(sys.argv) >= 2 and sys.argv[1] == "emb":
    words = []
    while True:
        try:
            word = input('word: ')
            
            if word == 'exit':
                # f.close()
                exit()
            elif word == 'save':
                filen = input('file name: ')
                f = open('{}.txt'.format(filen),'w',encoding='utf-8')
                for w_ in words:
                    f.write('%s\t%s\n' % (w_[0],str(w_[1])))
                words = []
                f.close()
                continue
            elif word == 'make_gen':
                
                gen = ['레즈비언','똥까시','섹파','몸팔','질싸','발정','애무','애무하','딸딸이','싼다','한남충','한녀충','노콘','썅년아','꽃뱀','꼴페미','삼일한','된장녀','된장남','성괴','페미년/N','스시녀/N','자지/N','게이/N','김치남/N','김치녀/N','꼴린다/V','매춘/N','발기/N','발기하/V','발기해/V','사정하/V','섹스하/V','스시녀/N','싼다/V','아다/N','좆물받이/N','좆물통/N','창녀/N','좆/N','좆방망이/N','보지/N']#,'강간하/V']#,'성추행/N','성추행하/V']#,'강간하/V','강간했/V']
                # gen = ['여혐','남혐','자지','게이','김치남','김치녀','꼴린다','매춘','발기','발기하','발기해','사정하','섹스하','스시녀','싼다','아다','좆물받이','좆물통','창녀','페미니스트','강간','좆','좆방망이','보지','강간하','성추행','성추행하']#,'강간하','강간했']
                # gen = ['김치남/N']
                for w in gen: 
                    w = w.replace('/N','').replace('/V','')
                    try:
                        sim_w = model.wv.most_similar_cosmul(w,topn=30)
                    except Exception as ex:
                        print(str(ex))
                        continue
                    words.append(['#'+w,0.0])
                    print(sim_w)
                    sim_ = []
                    for sw in sim_w:
                        words.append(sw)
                        sim_.append(sw[1])
                    avg_sim = sum(sim_)/len(sim_)
                    avg_sim = int(avg_sim * 100.0)/100.0
                    print(avg_sim)
                    if len(words) == 0:
                        continue
                    f = open('dictionary/gender/{}_1st.txt'.format(w.replace('/N','').replace('/V','')),'w',encoding='utf-8')
                    for w_ in words:
                        if w_[1] < (avg_sim) or len(w_[0]) == 1:#(avg_sim-0.1):
                            continue
                        f.write('%s\t%s\n' % (w_[0],str(w_[1])))
                    words = []
                f.close()
                exit()
            elif word == 'make_soc':
                
                # gen = ['여혐/N','남혐/N','자지/N','게이/N','김치남/N','김치녀/N','꼴린다/V','매춘/N','발기/N','발기하/V','발기해/V','사정하/V','섹스하/V','스시녀/N','싼다/V','아다/N','좆물받이/N','좆물통/N','창녀/N','페미니스트/N','강간/N','좆/N','좆방망이/N','보지/N','강간하/V','성추행/N','성추행하/V']#,'강간하/V','강간했/V']
                # gen = ['여혐','남혐','자지','게이','김치남','김치녀','꼴린다','매춘','발기','발기하','발기해','사정하','섹스하','스시녀','싼다','아다','좆물받이','좆물통','창녀','페미니스트','강간','좆','좆방망이','보지','강간하','성추행','성추행하']#,'강간하','강간했']
                # gen = ['김치남/N']
                gen = ['탄핵/N','탄핵하/V','각하/N','북중원/N','더민주/N','문재앙/N','홍준표/N','홍어/N','노무현/N','보수/N','진보/N','우익/N','좌익/N','사회주의/N','빨갱이/N','북한/N','김정은/N','김정일/N','개돼지들/N','박근혜/N','닭그네/N','닭년/N','이명박/N','쥐박/N','간첩/N','개누리당/N','친일파/N','친일/N','좌빨/N','우빨/N','개누리당/N','매국노/N','개대중/N','좌좀/N','우좀/N']
                for w in gen: 
                    w = w.replace('/N','').replace('/V','')
                    try:
                        sim_w = model.wv.most_similar_cosmul(w,topn=40)
                    except Exception as ex:
                        print(str(ex))
                        continue
                    words.append(['#'+w,0.0])
                    print(sim_w)
                    sim_ = []
                    for sw in sim_w:
                        words.append(sw)
                        sim_.append(sw[1])
                    avg_sim = sum(sim_)/len(sim_)
                    avg_sim = int(avg_sim * 100.0)/100.0
                    print(avg_sim)
                    if len(words) == 0:
                        continue
                    f = open('dictionary/soc/{}_1st.txt'.format(w.replace('/N','').replace('/V','')),'w',encoding='utf-8')
                    for w_ in words:
                        if w_[1] < (avg_sim) or len(w_[0]) == 1:
                            continue
                        f.write('%s\t%s\n' % (w_[0],str(w_[1])))
                    words = []
                f.close()
                exit()
            elif word == 'make_toxic':
                
                # gen = ['여혐/N','남혐/N','자지/N','게이/N','김치남/N','김치녀/N','꼴린다/V','매춘/N','발기/N','발기하/V','발기해/V','사정하/V','섹스하/V','스시녀/N','싼다/V','아다/N','좆물받이/N','좆물통/N','창녀/N','페미니스트/N','강간/N','좆/N','좆방망이/N','보지/N','강간하/V','성추행/N','성추행하/V']#,'강간하/V','강간했/V']
                # gen = ['여혐','남혐','자지','게이','김치남','김치녀','꼴린다','매춘','발기','발기하','발기해','사정하','섹스하','스시녀','싼다','아다','좆물받이','좆물통','창녀','페미니스트','강간','좆','좆방망이','보지','강간하','성추행','성추행하']#,'강간하','강간했']
                # gen = ['김치남/N']
                gen = ['탄핵/N','탄핵하/V','각하/N','북중원/N','더민주/N','문재앙/N','홍준표/N','홍어/N','노무현/N','보수/N','진보/N','우익/N','좌익/N','사회주의/N','빨갱이/N','북한/N','김정은/N','김정일/N','개돼지들/N','박근혜/N','닭그네/N','닭년/N','이명박/N','쥐박/N','간첩/N','개누리당/N','친일파/N','친일/N','좌빨/N','우빨/N','개누리당/N','매국노/N','개대중/N','좌좀/N','우좀/N']
                gen = ['병신','ㅅㅂ','시발','개같은놈','개같은년','개새끼','ㅅㄲ','새끼','좆까','시팔','미친놈','미친','ㅁㅊ','일베','개돼지','개돼지들','애미애비','애미','애비','꼬라지','느금마','시발놈아','운지','존나','ㅈㄴ','쓰레기팀','쓰레기','쓰레기새끼']
                for w in gen: 
                    w = w.replace('/N','').replace('/V','')
                    try:
                        sim_w = model.wv.most_similar_cosmul(w,topn=40)
                    except Exception as ex:
                        print(str(ex))
                        continue
                    words.append(['#'+w,0.0])
                    print(sim_w)
                    sim_ = []
                    for sw in sim_w:
                        words.append(sw)
                        sim_.append(sw[1])
                    avg_sim = sum(sim_)/len(sim_)
                    avg_sim = int(avg_sim * 100.0)/100.0
                    print(avg_sim)
                    if len(words) == 0:
                        continue
                    f = open('dictionary/toxic/{}_1st.txt'.format(w.replace('/N','').replace('/V','')),'w',encoding='utf-8')
                    for w_ in words:
                        if w_[1] < (avg_sim) or len(w_[0]) == 1:
                            continue
                        f.write('%s\t%s\n' % (w_[0],str(w_[1])))
                    words = []
                f.close()
                exit()
            elif word == 'make_age':
                
                # gen = ['여혐/N','남혐/N','자지/N','게이/N','김치남/N','김치녀/N','꼴린다/V','매춘/N','발기/N','발기하/V','발기해/V','사정하/V','섹스하/V','스시녀/N','싼다/V','아다/N','좆물받이/N','좆물통/N','창녀/N','페미니스트/N','강간/N','좆/N','좆방망이/N','보지/N','강간하/V','성추행/N','성추행하/V']#,'강간하/V','강간했/V']
                # gen = ['여혐','남혐','자지','게이','김치남','김치녀','꼴린다','매춘','발기','발기하','발기해','사정하','섹스하','스시녀','싼다','아다','좆물받이','좆물통','창녀','페미니스트','강간','좆','좆방망이','보지','강간하','성추행','성추행하']#,'강간하','강간했']
                # gen = ['김치남/N']
                # gen = ['탄핵/N','탄핵하/V','각하/N','북중원/N','더민주/N','문재앙/N','홍준표/N','홍어/N','노무현/N','보수/N','진보/N','우익/N','좌익/N','사회주의/N','빨갱이/N','북한/N','김정은/N','김정일/N','개돼지들/N','박근혜/N','닭그네/N','닭년/N','이명박/N','쥐박/N','간첩/N','개누리당/N','친일파/N','친일/N','좌빨/N','우빨/N','개누리당/N','매국노/N','개대중/N','좌좀/N','우좀/N']
                gen = ['개중딩','개고딩','노키즈존','김여사','개초딩','룸나무','휴먼급식체','초글링','맘카페','맘충','중2병','초딩','중딩','고딩','급식충','틀딱','틀니','개초딩','좆중딩','좆고딩','학식충','잼민이','할매미','연금충','꼰대','노슬아치','틀딱충','쉰내','실버타운','소년원','일진','일진놀이','삥뜯기','삥','빵셔틀','왕따','은따','찐따','민짜','미짜','꽌짝','노친네','히키코모리']
                for w in gen: 
                    w = w.replace('/N','').replace('/V','')
                    try:
                        sim_w = model.wv.most_similar_cosmul(w,topn=40)
                    except Exception as ex:
                        print(str(ex))
                        continue
                    words.append(['#'+w,0.0])
                    print(sim_w)
                    sim_ = []
                    for sw in sim_w:
                        words.append(sw)
                        sim_.append(sw[1])
                    avg_sim = sum(sim_)/len(sim_)
                    avg_sim = int(avg_sim * 100.0)/100.0
                    print(avg_sim)
                    if len(words) == 0:
                        continue
                    f = open('dictionary/age/{}_1st.txt'.format(w.replace('/N','').replace('/V','')),'w',encoding='utf-8')
                    for w_ in words:
                        if w_[1] < (avg_sim) or len(w_[0]) == 1:
                            continue
                        f.write('%s\t%s\n' % (w_[0],str(w_[1])))
                    words = []
                f.close()
                exit()
            sim_w = model.wv.most_similar_cosmul(word,topn=100)
            words.append(['#'+word,0.0])
            print(sim_w)
            for sw in sim_w:
                words.append(sw)
            # f.write(word+'\n')
            # for sw in sim_w:
            #     f.write(sw[0]+'\n')
                
        except Exception as ex:
            print(str(ex))

if len(sys.argv) >= 2 and sys.argv[1] == "2nd":
    f = open('1st.txt',encoding='utf-8')
    tot_score = 0.0
    cnt = 0
    for l in f:
        if l.startswith('#'):
            continue
        l = l.split('\t')[1].strip()
        score = float(l)
        cnt += 1
        tot_score += score
    avg_score = tot_score/cnt
    avg_score = avg_score + (avg_score * 0.2)
if False:
    #성혐오 결과
    seed = ['김치녀'
    # ,'김치남'
    # ,'스시남'
    ,'페미'
    ,'페미니즘'
    ,'동성애'
    ,'성혐오'
    ,'여혐'
    ,'남혐'
    ,'된장녀'
    # ,'된장남'
    ,'혐오'
    ,'걸레갈보개'
    ,'김치맨'
    ,'구멍동서'
    ,'보지'
    ,'자지'
    ,'좆물받이'
    ,'좆물'
    ,'보지'
    ,'보지년'
    ,'김치년'
    ,'후장'
    ,'창녀'
    ,'창놈']

    seed = ['성소수자']

    resultSeedF = open('result1stDict.txt','w',encoding='utf-8')

    for sd in seed:
        re_sim_word = model.wv.most_similar(sd)
        resultSeedF.write('#'+sd+'\n')
        for sim_word in re_sim_word:
            resultSeedF.write(sim_word[0]+'\t'+str(sim_word[1])+'\n')

    resultSeedF.close()

def word_most_sim(wordList,file_pre):
    words = {}
    resultWords = []
    for word in wordList:
        simwords = model.wv.most_similar(word)

        resultWords.append('#'+word)
        for simword_ in simwords:
            if simword_ in words:
                continue
            words[simword_[0]] = 1
            resultWords.append(simword_[0])

    resultSeedF = open('resultDict_'+file_pre+'.txt','w',encoding='utf-8')
    resultSeedF.write('\n'.join(resultWords)+'\n')
    resultSeedF.close()

def make_dic(file_pre):
    with open('dictionary/'+file_pre+'/extend/dictionary_'+file_pre+'.txt',encoding='utf-8') as dic_word:
        word_count = 0
        seed_dic = np.zeros(300)
        _duplicate = {}
        for word in dic_word:
            if word.startswith('//'):
                continue
            
            word = word.replace('#','').strip().split('\t')
            
            if word[0] in _duplicate:
                continue
            
            try:
                seed_dic += model.wv[word[0]]
            except:
                continue
            word_count += 1
            _duplicate[word[0]] = True
            # print(word)
            # print(model.wv[word[0]])
            
        seed_dic /= word_count
        np.save('emb/'+file_pre,seed_dic)
# make_dic('age')
# exit()
make_dic('gender')
make_dic('soc')
make_dic('toxic')


ageList = [
    '틀딱'
    ,'박사모'
    ,'꼰대'
    ,'초딩'
    ,'중딩'
    ,'고딩'
    ,'급식충'
    ,'좆고딩'
    ,'노인네'
    ,'노친네'
    ,'늙은이'
    ,'닭사모'
    ,'관짝'
    ,'틀니'
    ,'노인네'
    ,'산송장'
    ,'틀니충'
    ,'할배'
    ,'할매'
    ,'아줌마'
    ,'아재'
    ,'늙은이'
    ,'검버섯'
    ,'애미'
    ,'애비'
    ,'노총각'
    # ,'씹틀딱'
]

# word_most_sim(ageList,'age')
# make_dic('age')
# import sys
# sys.exit()
locationList = [
    '홍어'
    ,'홍어새끼들'
    ,'탈라도'
    ,'홍어들'
    ,'경상'
    ,'경상도'
    ,'지역감정'
    ,'전라디언'
    ,'절라도'
    ,'탈라도'
    ,'국민의당'
    ,'쌍도'
    ,'개쌍도'
    ,'대구놈들'
    ,'홍어천지'
    ,'촌년'
    ,'촌놈'
    ,'탐라국'
    ,'강정마을'
    ,'탐라'
    ,'홍어밭'
    ,'감자국'
    ,'설라디언'
]

if False:
    wordList = [
        '좌좀'
        ,'좌빨'
        ,'좌좀'
        ,'우파'
        ,'좌파'
        ,'빨갱이'
        ,'친북'
        ,'극우'
        ,'간첩'
        ,'보수'
        ,'진보'
        ,'매국노'
        ,'친일'
        ,'좌빨'
        ,'김정일'
        ,'김정은'
    ]

    words = {}
    resultWords = []
    for word in wordList:
        simwords = model.wv.most_similar(word)

        resultWords.append('#'+word)
        for simword_ in simwords:
            if simword_ in words:
                continue
            words[simword_[0]] = 1
            resultWords.append(simword_[0])

    resultSeedF = open('resultDict_soc.txt','w',encoding='utf-8')
    resultSeedF.write('\n'.join(resultWords)+'\n')
    resultSeedF.close()

word_count = 0
seed_dic = np.zeros(300)
_duplicate = {}
with open('dictionary/gender/extend/dictionary_gender.txt',encoding='utf-8') as dic_word:
    for word in dic_word:
        if word.startswith('//'):
            continue
        # print(word)
        word = word.replace('#','').strip().split('\t')
        
        if word[0] in _duplicate or word[0].strip() == '':#'/N' or word[0].strip() == '/V':
            continue

        try:
            seed_dic += model.wv[word[0]]
            word_count += 1
            _duplicate[word[0]] = True
        except Exception as ex:
            print(ex)

with open('dictionary/soc/extend/dictionary_soc.txt',encoding='utf-8') as dic_word:

    _duplicate = {}
    for word in dic_word:
        if word.startswith('//'):
            continue
        
        word = word.replace('#','').strip().split('\t')
        
        if word[0] in _duplicate or word[0].strip() == '':#'/N' or word[0].strip() == '/V':
            continue

        # print(word)
        # print(model.wv[word[0]])
        try:
            seed_dic += model.wv[word[0]]           
            word_count += 1
            _duplicate[word[0]] = True
        except Exception as ex:
            print(ex)
if False:    
    with open('dictionary/age/extend/dictionary_age.txt',encoding='utf-8') as dic_word:
        # word_count = 0
        # seed_dic = np.zeros(300)
        _duplicate = {}
        for word in dic_word:
            if word.startswith('//'):
                continue
            
            word = word.replace('#','').strip().split('\t')
            
            if word[0] in _duplicate or word[0].strip() == '':#'/N' or word[0].strip() == '/V':
                continue
                
            word_count += 1
            # print(word[0])
            _duplicate[word[0]] = True
            # print(word)
            # print(model.wv[word[0]])
            seed_dic += model.wv[word[0]]
seed_dic /= word_count
np.save('emb/total',seed_dic)
total = seed_dic
print(total)
gen = np.load('emb/gender.npy')
soc = np.load('emb/soc.npy')
txc = np.load('emb/toxic.npy')
age = np.load('emb/age.npy')
tt = np.load('emb/total.npy')

word = {}
word['[PAD]'] = 0
word['[UNK]'] = 1
# word['']
wordindex = 2
import csv
import re
csvf = open('dataset.csv','w',encoding='utf-8', newline='')
wr = csv.writer(csvf)

def labeling(sim):
    if sim < 0.5:
        return 0
    else:
        return 1

with open('preprocessing/ht_x.txt',encoding='utf-8') as htx:
    with open('preprocessing/ht_origin.txt',encoding='utf-8') as htori:
        with open('preprocessing/ht_origin_v.txt',encoding='utf-8') as htori2:
            for l, o, o2 in zip(htx,htori,htori2):
                l = l.strip()
                l = re.sub(' +',' ',l)
                l = l.split(' ')

                o = o.strip()
                o2 = o2.strip()
                if o == '' or o2 == '' or l == '':
                    # print('1',o)
                    continue
                o = re.sub(' +',' ',o)
                o = o.split(' ')

                o2 = re.sub(' +',' ',o2)
                o2 = o2.split(' ')

                temp_nouns = []
                for noun in o2:
                    try:
                        if noun not in word:
                            word[noun] = wordindex
                            wordindex += 1

                        temp_nouns.append(model.wv[noun])
                        
                    except Exception as ex:
                        a = 1
                _worddic = np.array(temp_nouns)

                if _worddic.shape[0] == 0:
                    continue
                    # print(np.sum(_worddic,axis=0).shape,l,len(temp_nouns))
                _worddic = np.sum(_worddic,axis=0) / len(temp_nouns)

                if len(temp_nouns) == 0:
                    # _worddic = np.zeros(300)
                    continue


                wr.writerow([' '.join(l),' '.join(o),labeling(cos_sim(total,_worddic)),labeling(cos_sim(gen,_worddic)),labeling(cos_sim(soc,_worddic)),labeling(cos_sim(txc,_worddic)),labeling(cos_sim(age,_worddic))])
csvf.close()