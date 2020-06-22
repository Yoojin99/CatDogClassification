# CatDogClassification :cat::dog:
개 vs 고양이 분류 CNN 모델 학습 연습

***
참고한 사이트 : <https://provia.tistory.com/79>
수정사항 : "keras.-" 형태로 모듈을 import 하면 에러 발생하여 "tensorflow.keras.-"형태로 모듈을 import하는 것으로 수정하였다.
***

## 이용한 데이터 셋

kaggle Dogs vs. Cats Dataset
<https://www.kaggle.com/c/dogs-vs-cats/data>

## 파일 설명

CatDogClassification.ipynb -> cat dog classify 하는 모델 학습 및 weight 파일 저장하는 코드
evaluation.ipynb -> 모델 테스트 코드
model.h5 -> 학습시킨 모델의 가중치 파일

## 모델 테스트 방법

시작하기 전에...
1. model의 weight파일이 다운받아져 있어야 한다. (model.h5)
2. 테스트할 데이터들이 있는 폴더가 존재해야 한다.

#### 0. Test 준비하기
~~~
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from os import listdir
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
#print(os.listdir(r"C:\Users\YooJin\Desktop\topcit\data")) <하위폴더가 있을 경우, 하위폴더들을 출력한다.
~~~

#### 1. 기본 변수 선언
~~~
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
~~~

#### 2. 모델 예측
학습한 모델로 1.에서 생성한 test 셋을 넣는다.
~~~
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
~~~

#### 3. 모델 불러오기 위한 함수
모델을 불러오기 위해 모델의 뼈대를 만드는 create_model() 함수를 만든다. 후에 여기에 저장된 weight를 불러올 것이다.
~~~
def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
~~~

#### 4. 데이터 준비
prediction의 결과는 각 record 별 개일 확률 ~, 고양이일 확률 ~ 로 출력된다.
개와 고양이일때의 확률을 비교하여 더 크게 나타난 쪽으로 레이블을 선택해서 값을 치환한다.
~~~
test_filenames = os.listdir("C:\\Users\\YooJin\\Desktop\\topcit\\data\\test1\\") #여기서 원하는 디렉토리(폴더) 지정
test_df = pd.DataFrame({
    'filename' : test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
  test_df,
  "C:\\Users\\YooJin\\Desktop\\topcit\\data\\test1\\", #경로 재지정 필요
  x_col='filename',
  y_col=None,
  class_mode=None,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
  shuffle=False
)
~~~

#### 4. 모델 load
~~~
model = create_model()
model.load_weights("C:\\Users\\YooJin\\Desktop\\topcit\\model.h5") #모델 load, 경로 재지정 필요
~~~

#### 5. 데이터 예측 및 결과 확인
~~~
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size)) #데이터 예측

test_df['category'] = np.argmax(predict, axis=-1)
test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(20)
sample_test.head()
plt.figure(figsize=(12,24))
for index, row in sample_test.iterrows():
  filename = row['filename']
  category = row['category']
  img = load_img("C:\\Users\\YooJin\\Desktop\\topcit\\data\\test1\\"+filename, target_size=IMAGE_SIZE) #경로 재지정 필요
  plt.subplot(5,4,index+1)
  plt.imshow(img)
  plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()

#개가 1, 고양이가 0으로 나오면 정답이다.
~~~
