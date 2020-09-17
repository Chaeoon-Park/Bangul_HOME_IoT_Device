import RPi.GPIO as GPIO #RPi.GPIO 라이브러리를 GPIO로 사용
from time import sleep  #time 라이브러리의 sleep함수 사용
'''
서보 위치 제어 함수
degree에 각도를 입력하면 duty로 변환후 서보 제어(ChangeDutyCycle)
'''
def camera_motor_control(now_angle, rotate_angle) :
    # 각도는 180도를 넘을 수 없다.
    if rotate_angle < 10 :
        return now_angle


    servoPin          = 17   # 서보 핀
    SERVO_MAX_DUTY    = 12.8  # 서보의 최대(180도) 위치의 주기
    SERVO_MIN_DUTY    = 3.2    # 서보의 최소(0도) 위치의 주기

    GPIO.setmode(GPIO.BCM)        # GPIO 설정
    GPIO.setup(servoPin, GPIO.OUT)  # 서보핀 출력으로 설정

    servo = GPIO.PWM(servoPin, 50)  # 서보핀을 PWM 모드 50Hz로 사용하기 (50Hz > 20ms)
    servo.start(0)  # 서보 PWM 시작 duty = 0, duty가 0이면 서보는 동작하지 않는다.

    degree = now_angle + rotate_angle
    if degree > 180:
        degree = 180
    if degree < 0 :
        degree = 0

    # 각도(degree)를 duty로 변경한다.
    duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
    # duty 값 출력
    print("Degree: {} to {}(Duty)".format(degree, duty))

    # 변경된 duty값을 서보 pwm에 적용
    servo.ChangeDutyCycle(duty)

    # 모터의 다음 입력까지 꼭 필요한 시간. 이동안은 검출도 하지 않도록 한다. 쓰레드 처리시 더 정확도가 떨어짐
    sleep(0.25)

    GPIO.cleanup()
    return degree
