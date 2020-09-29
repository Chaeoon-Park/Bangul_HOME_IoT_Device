# Bangul_HOME_IoT_Device
"차량용 스마트 펫케어 서비스 : 방울이가타고있어요"의 HOME_Device [강아지 추적용 카메라] 입니다

## 관련 Repository
- FE : https://github.com/jikjoo/Bangul_webOS
- 서버 : https://github.com/Galocg/Bangul_Smart
- 머신러닝 : https://github.com/Adam-Kim/dogvomitML
- 디자인 :https://github.com/jikjoo/moonstone
- webRTC : https://github.com/jikjoo/react-webrtc

## 설치
- 본 기기는 HW로 Raspberry Pi 4(OS : Raspbian), Google Coral USB (선택사항 , 연산속도를 크게 늘려줍니다.), Servo Motor, Pi Camera를 필요로 합니다.
- 본 기기는 Object Detection SW Tool 로 TFlite와 동봉된 Sample 모델을 사용합니다. 
- TFlite와 Google Coral USB를 통한 TPU장치의 사용은 https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md#part-1---how-to-set-up-and-run-tensorflow-lite-object-detection-models-on-the-raspberry-pi 를 참고하여 설치 합니다. (Open CV의 설치를 포함합니다)
- git clone `https://github.com/Chaeoon-Park/Bangul_HOME_IoT_Device` 의 src 폴더 내부의 파일을 Raspberry Pi 4의 tflite1 폴더에 풀어놓습니다.

## 사용
 * 폴더 이동 
 cd /home/pi/tflite1
 * 가상 환경 접속
 source tflite1-env/bin/activate
 * 실행
 python3 bangul_home_iot.py --modeldir=Sample_TFLite_model --edgetpu


## 실행 영상


----------------------------
# 관련 문서

- 실행 로직
- 객체의 Depth를 구하기
- 추적 알고리즘 : 칼만필터, 헝가리안 알고리즘, 전용 알고리즘
- TFlite와 TPU 연산장치
