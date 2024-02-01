# 목차
1. [프로젝트 주제](#프로젝트-주제)
2. [프로젝트 목표](#프로젝트-목표)
3. [개발 환경](#개발-환경)
4. [개발 기간 및 수행 절차](#개발-기간-및-수행-절차)
5. [데이터 제작](#데이터-제작)
6. [모델 학습](#모델-학습)
7. [하이퍼 파라미터](#하이퍼-파라미터)
8. [Code](#Code)
9. [훈련 결과](#훈련-결과)
10. [프로젝트 결과](#프로젝트-결과)
11. [프로젝트 후기](#프로젝트-후기)
12. [참조](#참조)
13. [제출물](#제출물)
<br>
<br>

## 대회 링크

https://ubai.uos.ac.kr/forum/view/767800

<br>

## 프로젝트 주제
- 토지특성 파악을 위한 인공지능 알고리즘 구축이 지정 주제였으며 3개의 소주제 중 하나를 선택하여 모델을 개발하는 것이 목표
- 그 중에서 '딥러닝 또는 머신러닝을 통한 토지특성 정보 구축(지형 형상)'을 선택하였으며 대상지로는 동대문구를 선정


<br>

## 프로젝트 목표
<div align='center'>
<img src='https://github.com/ha789ha/lx_ubai/assets/108510929/94e401f5-d314-4a53-b396-47b98434cda9'>
</div>
<br>

- 국토교통부에서는 표준지 토지특성 조사요령에서 지형 형상의 구분 기준을 제시하고 있음
- 이를 AI모델을 통해 수기가 아닌 자동으로 지형 형상을 구분해주는 모델을 개발하는 것이 목표
- 이를 위해 객체 탐지 종류의 하나인 instance segmentation 모델을 사용하기로 하였으며 그 중에서 1-stage detector로 돼있어 속도가 빠르고 비교적 정확도가 높은 YoloV5 모델을 사용

<br>

## 개발 환경
- Deeplearning Model : [YoloV5](https://github.com/ultralytics/yolov5)
- 데이터 제작: Rhino6
- 버전 및 이슈관리 : Github
- OS : Windows 10
- CPU : 12th Gen Intel(R) Core(TM) i9-12900K
- RAM : 64.0GB
- GPU : NVIDIA GeForce RTX 3090 Ti
- CUDA : 11.2
- cudnn : 8.2
- Python : 3.7.16
- torch : 1.8.0
- torchvision : 0.9.0
- detectron2 : 0.2.1 (detectron2-windows)
<br>

## 개발 기간 및 수행 절차
- 전체 개발 기간 : 2023-01-12 ~ 2023-02-28
<br>

## 데이터 제작
<div align='center'>
  
|**원본 데이터**|
| :---: |
|<img width="464" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/8038fefb-947d-481d-8343-c4260f5fff54">|

</div>


- 토지형상의 데이터는 위와 같이 Shp파일로 돼있었으며 nstance segmentation에 훈련 데이터로 사용하기 위해서는 가공을 통해 구역을 나눌 필요 존재
- 이를 위해 Shp 파일 편집 프로그램인 Rhino6를 사용
<br>

<div align='center'>
  
|**가공 데이터**|
| :---: |
|<img width="800" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/aa920396-5242-4293-838d-41192379f3ae">|
</div>

- 다음과 같이 2,3개씩 그룹을 묶어 이미지를 제작하였으며 yolov5 labeling을 위해 우측의 이미지와 같이 색을 통해 구분
- 또한 좌측의 이미지와 같이 색을 없앤 훈련 데이터 또한 쌍으로 생성

<br>

## 데이터 전처리
<div align='center'>
<img width="500" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/c68fbf41-641a-4c17-bc16-8865c33f2622">
</div>

  1. labelig
    본격적인 라벨링을 위해서 cv2 라이브러리의 findContour 함수를 사용하여 외곽선을 검출, 그 후 Teh-Chin 근사 알고리즘을 적용하여 클래스 별로 굴곡점의 포인트를 추출하고 좌표를 txt파일에 저장

  2. train/test split
    train-test split 함수를 통해 8:1:1의 비율로 학습, 검증, 평가 데이터셋 분할
<br>

## 모델 학습
- 학습에 사용한 yolov5모델은 [ultralytics](https://github.com/ultralytics/yolov5)의 소스를 사용하였으며 이 때, 모델 학습 시 사용할 데이터의 경로, 클래스를 정의하기 위한 yaml파일을 생성

<br>

## 하이퍼 파라미터
<img width="563" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/81316ee8-34b0-4a39-b455-e9af86e83444">


## Code
1. download data: https://drive.google.com/file/d/13tlKbZRhfLQ_SOrBWIN6soFA6wgjk5Jn/view?usp=drive_link
   
  
2. Clone this repository:
```bash
git clone https://github.com/ha789ha/lx_ubai.git
cd lx_ubai
```

3. generate label:
```bash
python label_generator.py
```

4. data split:
```bash
python data_split.py
```

5. download yolov5:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip instal -r requirements.txt
```

6. training:
```bash
python train.py --img 640 --epochs 300 --batch-size 16 --data lx_ubai/dataset.yaml --weights yolov5s.pt --lr 0.01 --final-lr 0.1 --momentum 0.937 --weight-decay 0.0005 --warmup-epochs 3.0 --warmup-momentum 0.8 --warmup-bias-lr 0.1
```
<br>

## 훈련 결과
- 정성적 평가

|**평가 데이터 정답 레이블**|**평가 데이터 예측 결과**|
| :---: | :---: |
|<img width="450" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/a1afb472-e270-45db-b44e-2449cc17c76a">|<img width="447" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/615c2804-58ab-4bac-bac8-045a261ece61">|

- 정량적 평가

<img width="610" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/0d9230a7-ff8c-46ba-9357-d66169ce8a45">

- confusion matrix

<img width="700" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/d3df9d70-6618-478a-a9f5-c8e33c29aa4e">

<br>

## 프로젝트 결과
전체적으로 Precision 0.565, Recall 0.720을 기록, 전체적인 성능의 수준이 좋지 않아 분석한 결과, 크게 두 가지 원인이 존재

<h3>1. 국토교통부에서 정한 기준 자체에서 오류 존재</h3>
<div align='center'>
<img width="400" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/fda8c0ad-56d7-488d-bad9-671b78a7a764">
</div>
<br>
위 사진은 '세로장방형'으로 분류된 객체이나 토지의 넒은 면이 도로와 접하고 있어 '가로장방형'에 가까움

<h3>2. 분류 기준의 모호함</h3>
<div align='center'>
<img width="400" alt="image" src="https://github.com/ha789ha/lx_ubai/assets/108510929/7a995133-21cf-42c3-8f95-b58cf389be26">
</div>
<br>
위의 객체는 '정방형'으로 분류된 객체, 국토교통부에서는 '장방형'과 '정방형'에 대해 구분하고 있는데 내각을 보면 '정방형'보다는 '장방형'에 가까운 것을 알 수 있음

## 프로젝트 후기
위의 결과를 토대로 AI 모델의 결과의 아쉬움에 대해 원인을 분석하였고 분류 기준에 대한 중요성을 역설하였습니다. 이는 많은 심사위원들의 공감을 사 대회를 1등으로 마무리 할 수 있었습니다.<br>
이전 Dacon이나 Kaggle 같은 대회에서는 잘 정제된 데이터를 토대로 모델 성능을 끌어올리는 데에만 집중했다면 이번 공모전을 통해 데이터 제작, 전처리, 모델 훈련까지 end to end로 경험할 수 있었고 데이터의 중요성을 느낄 수 있는 공모전이었습니다.

## 참조
- [국토교통부 2023년도 표준지공시지가 조사평가 업무요령](http://www.molit.go.kr/USR/policyData/m_34681/dtl.jsp?id=4644)
- [YOLOV5](https://github.com/ultralytics/yolov5)


## 제출물
자세한 내용은 아래의 보고서를 참고해주시면 감사하겠습니다.
<br>
[공간 정보 활용 공모전 보고서](https://github.com/ha789ha/lx_ubai/files/14123467/default.pdf)

