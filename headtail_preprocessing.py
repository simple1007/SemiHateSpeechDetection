import re

tag = ['N','V']
ht_nv = open('preprocessing/ht_nv.txt','w',encoding='utf-8')
ht_n = open('preprocessing/ht_n.txt','w',encoding='utf-8')
with open('rawdata/ilbe_result.txt',encoding='utf-8') as origin:
    with open('preprocessing/ilbe_morph.txt',encoding='utf-8') as ht:
        for ht_, origin_ in zip(ht,origin):
            ht_te = ht_.strip()
            ht_te = re.sub(' +',' ',ht_te)

            ht_te = ht_te.split(' ')
            nv_temp = []
            n_temp = []
            for ht_te_ in ht_te:
                ht_te_ = ht_te_.split('+')

                for ht_te__ in ht_te_:
                    ht_te__ = ht_te__.split('/')
                    if ht_te__[1][0] in tag:
                        nv_temp.append(ht_te__[0])
                    if ht_te__[1][0] == 'N':
                        n_temp.append(ht_te__[0])
            
            if len(nv_temp) > 0:
                ht_nv.write(' '.join(nv_temp)+'\n')
            if len(n_temp) > 0:
                ht_n.write(' '.join(n_temp)+'\n')

ht_nv.close()
ht_n.close()