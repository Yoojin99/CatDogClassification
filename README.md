# CatDogClassification :cat::dog:
개 vs 고양이 분류 CNN 모델 학습 연습

***
참고한 사이트 : <https://provia.tistory.com/79>
***

## 이용한 데이터 셋


## 모델 테스트 방법

시작하기 전에...
1. 바로 "keras.~" 형태로 모듈을 import 하면 에러 발생하여 "tensorflow.keras.~"형태로 모듈을 import
2. model이란 변수에 model load 함
3. test 전 선행되어야 하는 부분
'''
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import maplotlib.pyplot as plt
import os

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

batch_size=15

'''

#### 0. Test 시작하기
'''
test_filenames = os.listdir("학습할 데이터들이 있는 폴더")
test_df = pd.DataFrame({
  'filename': test_filenames
})
nb_samples = test_df.shape[0]
'''

#### 1. 평가 데이터 준비
'''
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
  test_df,
  "(data의 test1이 위치하는 경로 ex ./data/test1/)",
  x_col='filename',
  y_col=None,
  class_mode=None,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
  shuffle=False
)
'''

#### 2. 모델 예측
학습한 모델로 1.에서 생성한 test 셋을 넣는다.
'''
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
'''

#### 3. 평가 생성
prediction의 결과는 각 record 별 개일 확률 ~, 고양이일 확률 ~ 로 출력된다.
개와 고양이일때의 확률을 비교하여 더 크게 나타난 쪽으로 레이블을 선택해서 값을 치환한다.
'''
test_df['category'] = np.argmax(predict, axis=-1)
'''

#### 4. 레이블 변환
평가를 위해 dog, cat 형태로 되어 있는 데이터를 1, 0으로 변경한다.
'''
test_df['category'] = test_df['category'].replace({ 'dog':1, 'cat':0 })
'''

#### 5. 정답비율 확인
개와 고양이를 어느정도 비율로 예측했는지 본다.
'''
test_df['category'].value_counts().plot.bar()
'''

#### 6. 정답 확인
예측한 결과를 눈으로 확인한다.
'''
sample_test = test_df.head(18)
sample_test.head()
plit.figure(figsize=(12,24))
for index, row in sample_test.iterrows():
  filename = row['filename']
  category = row['category']
  img = load_img("test1의 경로 ex ./data/test1/"+filename, target_size=IMAGE_SIZE)
  plt.subplot(6,3,index+1)
  plt.imshow(img)
  plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
'''
