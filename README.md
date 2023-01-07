# SemiHateSpeechDetection
word2vec으로 데이터셋 구축 후 hatespeech detection 수행
1. 혐오성 발언이 발화되는 사이트의 댓글 데이터를 이용하여 word2vec을 수행한후
2. 유사도 비교로 각 클래스에 해당하는 어휘집(사전)을 구축
3. 구축된 사전의 문서 임베딩의 문서 임베딩과 댓글 문장 입베딩을 구성
4. 유사도 비교하여 특정 임계치를 넘겼을 경우 해당 클래스에 라벨 부여[ 클래스: 성혐오, 정치혐오, 욕설, 연령 혐오(댓글 데이터의 작성 일자가 예전것이라 현재와 비교하여 데이터 부족으로 부정확) ]
5. 모델의 output은 입력 문장이 각 라벨일 확률
* 추후 혐오 표현이 의심되는 단어 Detection 또한 수행 예정
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
