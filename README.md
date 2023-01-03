# SemiHateSpeechDetection
word2vec으로 데이터셋 구축 후 hatespeech detection 수행
## 디렉토리 구조
preprocessing / 형태소 분석되어진 텍스트 데이터셋
dictionary / 각클래스별 혐오 seed 어휘

## 1. Word2Vec 생성
```
python ilbe_word2vec.py
```

## 2. 각 클래스 별 어휘집 생성
```
python emb.py
```

## 3. dataset to numpy
프로젝트 루트에 ht 폴더생성
```
python ht_dataset.py
```

## 4. 학습 시작
```
python train.py
```

## 5. 모델 테스트
```
python inference_prob.py
```
