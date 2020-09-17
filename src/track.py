import bangul_matching
import math
from similarity import cosine_map 
import time
import collections as col

class detect_object :
    def __init__(self,label,number,value) :
        #검출 객체 종류
        self.label = label
        self.number = number
        #상태값 초기화 
        self.point = value['point'] #현재 좌표값
        self.color = value['color'] # 색
        self.gridbox = value['gridbox']
        self.timer = time.time() #감지된 시간
        self.detect = True

    def update(self, value ) :
        self.point = value['point']
        self.vector = 0 #벡터 갱신은 조끔 다른 문제인듯
        self.timer = time.time()
        self.color = value['color'] 
        self.gridbox = value['gridbox']
        self.detect = True

    # 확장형 칼만 필터 예측 및 수정 부분    
    # def ekf_predict(self) :
    # def ekf_correct(self) :


def rotate(exists,angle,detect_list) :
    for object_label in detect_list :
        exist_list = exists[object_label]
        LEN_EXIST = len(exist_list)
        for i in range(LEN_EXIST) :
            x,y = exist_list[i].point
            l = math.sqrt(pow(x,2) + pow(y,2))
            now_angle = (math.degrees(math.atan(y/x)) + angle) * 3.14 / 180 
            x = math.cos(now_angle) * l
            y = math.sin(now_angle) * l
            exist_list[i].point = [x,y]
            
    return exists
            

def track(exists,values,detect_list) :
    for object_label in detect_list :
        exist_list = exists[object_label] #얘는 class 
        value_list = values[object_label] #얘는 딕셔너리 형태
        LEN_EXIST = len(exist_list)
        LEN_NEW = len(value_list)

        #번호 지정
        deq = col.deque()
        numberlist = [False for i in range(11)] #최대 강아지의 수는 10마리로 지정
        for i in range(LEN_EXIST) :
            numberlist[exist_list.number] = True
        for i in range(1,11) :
            if numberlist[i] == False :
                deq.append(i)



        #color 맵 구성        
    
        exist_color = []
        for i in range(LEN_EXIST) :
            exist_color.append(exist_list[i].color)

        new_color = []
        for i in range(LEN_NEW) : 
            new_color.append(value_list[i]['color'])

        color_map = cosine_map(exist_color, new_color)

        #color 에 의한 헝가리안 매칭, 거리가 일정 수준 이하인 매칭의 경우 따로 빼줄 수 있도록 한다.


        #xy coordinate 맵 구성
        exist_point = []
        for i in range(LEN_EXIST) :
            exist_point.append(exist_list.point) #일단 현재 지점만 가지고 해보자
        new_point = []
        for i in range(LEN_NEW) : 
            new_point.append(value_list[i]['point'])

        xy_map = cosine_map(exist_point, new_point)
        #검출 된지 오래된 친구의 xy map의 거리는 일정 값으로 통일시킵니다.
        BASIC_DISTANCE = 3
        BASIC_TIME = 5
        for i in range(LEN_EXIST) :
            if exist_list[i].detect == False and time.time() - exist_list[i].timer > BASIC_TIME :
                for j in range(LEN_NEW) :
                    xy_map[i][j] = BASIC_DISTANCE


        #매칭을 실행한다
        update_exist = []
        match = bangul_matching.matching(color_map,xy_map)
        for past,now in match :
            if past == -1 :
                #새로운 객체를 생성한다
                exist_list.append(detect_object(object_label,deq.popleft(),value_list[now]))               
            else :
                #존재하던 객체를 갱신한다
                exist_list[past].update(value_list[now])
                update_exist.append(past)


        #현재 존재하는 객체는 False로 만든다.
        for i in range(LEN_EXIST) :
            if i not in update_exist:
                exist_list[i].detect = False

    return exists

    

