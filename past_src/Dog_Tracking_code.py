import math
import copy
import collections as col
import numpy as np



def Define_Angle(ymax,xmin,xmax) :
    #칼리브레이션 한 값을 이용해 Depth를 구하고 실제 각도를 구합니다.
    rx,ry= Change_Real_Parameter(xmin,ymax,xmax,ymax)
    real_angle = 0
    if ry>0 and rx<0 :
        real_angle  = math.atan2((-1 *rx) , ry)* (180/3.14) * -1
        real_angle = real_angle/2
    elif ry>0 and rx>=0 :
        real_angle  = math.atan2(rx , ry)* (180/3.14)
        real_angle = real_angle/2

    if real_angle <30 and (xmin+xmax)/2 < 200 :
        real_angle = -15
    elif  real_angle <30 and (xmin+xmax)/2 >1000 :
        real_angle = 15
        
    if abs(real_angle) < 15 :
        real_angle = 0
    
    return real_angle

#각 추적 객체를 위한 class 생성
class dog :
    def __init__(self,x,y,gridbox,number) :

        #강아지 번호
        self.number = number
        #Gridbox의 픽셀 좌표값
        self.ymin = gridbox[0]
        self.xmin = gridbox[1]
        self.ymax = gridbox[2]
        self.xmax = gridbox[3]
        self.x = (self.xmin + self.xmax) / 2
        self.y = self.ymax
        #강아지의 현재 벡터값 
        self.x_v = 0 
        self.y_v = 0 
       
        #새로 생성된 객체인지 판단
        self.newflag = True
    
        #칼만필터 계수
        self.Kalman_Gain = np.array([[0,0,0,0],[0,0,0,0] , [0,0,0,100] , [0,0,100,0] ]) #초기값 배열
        self.Kalman_P_now = np.array([[0,0,0,0],[0,0,0,0] , [0,0,0,100] , [0,0,100,0] ]) #초기값 배열     
        self.Kalman_P_predict = np.array([[0,0,0,0],[0,0,0,0] , [0,0,0,100] , [0,0,100,0] ])
        self.Kalman_z = np.array([[self.x],[self.y]])
        self.Kalman_x_now = np.array([[self.x],[self.x_v],[self.y],[self.y_v]])
        self.Kalman_x_predict = np.array([[self.x],[self.x_v],[self.y],[self.y_v]])

    # 칼만 필터를 통해 현재 객체의 예측값을 도출한다 인터벌은 마지막 예측시점부터 지금까지 걸린 시간.
    def KalmanFilter_Predict(self, interval) :
        A=np.array([[1, interval ,0,0],[0,1,0,0] , [0,0,1, interval] , [0,0,0,1] ]) 
        Q=np.array([[1,0,0,0],[0,1,0,0] , [0,0,1,0] , [0,0,0,1] ])
        self.Kalman_x_predict= A @ self.Kalman_x_now
        self.Kalman_P_predict = A @ self.Kalman_P_now @ np.transpose(A) + Q
        return [self.Kalman_x_predict[0][0],self.Kalman_x_predict[2][0]] #예측 좌표 도출

    
    # 매칭된 실제 좌표를 통해서 칼만 필터 계수를 갱신한다
    def KalmanFilter_Correct(self,correct) :
        correct_z_x, correct_z_y = correct
        H=np.array([[1,0,0,0],[0,1,0,0]])
        R=np.array([[50,0],[0,50]]) #R의 [1,1] , [2,2] 값을 조정함으로서 새로 들어온 벡터에 대한 민감도를 설정할 수 있다. 높을수록 예민함.
        I=np.array([[1,0,0,0],[0,1,0,0] , [0,0,1,0] , [0,0,0,1] ])
        self.Kalman_z = np.array([[correct_z_x],[correct_z_y]])
        self.Kalman_Gain =( self.Kalman_P_predict @ np.transpose(H) ) @ np.linalg.inv(H @ self.Kalman_P_predict @ np.transpose(H) + R )
        self.Kalman_x_now = self.Kalman_x_predict + self.Kalman_Gain @ (self.Kalman_z - H @ self.Kalman_x_predict)
        self.Kalman_P_now = (I- self.Kalman_Gain @ H ) @ self.Kalman_P_predict

    #상태값 갱신
    def updater(self, measured, match, interval, gridbox) :
            self.x_v = (measured[match[1]][0] - self.x) / interval
            self.y_v = (measured[match[1]][1] - self.y) / interval
            self.x = measured[match[1]][0]
            self.y = measured[match[1]][1]
            self.ymin = gridbox[match[1]][0]
            self.xmin = gridbox[match[1]][1]
            self.ymax = gridbox[match[1]][2]
            self.xmax = gridbox[match[1]][3]

global adj 
global visited,CoverA,CoverB
global match,matchx
def dfs(cur) :
    global adj 
    global visited,CoverA,CoverB    
    global match,matchx
    visited[cur] =True
    for to in adj[cur] :
        if matchx[to] == -1 or  (visited[matchx[to]] == False  and dfs(matchx[to])==True) :
            matchx[to]=cur
            match[cur]= to
            return True
    return False

def bfs(now) :
    global adj 
    global visited,CoverA,CoverB    
    global match,matchx
    deq = col.deque()
    deq.append(now)
    visited[now]= True
    CoverA[now] = False
    while len(deq)!=0 :
        cur = deq.popleft()
        CoverA[cur] = False
        for to in adj[cur] :
            if visited[matchx[to]]==True :
                continue
            if match[cur]!= to and matchx[to]!= -1 :
                deq.append(matchx[to])
                visited[matchx[to]]==True
                CoverB[to]= True
    return

#쾨니그 정리, 이분매칭을 사용한다.
def Konig(Map,n):
    global adj 
    global visited,CoverA,CoverB
    global match,matchx
    adj =  [[] for i in range(n)]
    match = [-1 for i in range(n)]
    matchx = [-1 for i in range(n)]
    CoverA = [True for i in range(n)]
    CoverB = [False for i in range(n)]
    for i in range(n) :
        for j in range(n) :
            if Map[i][j]==0 :
                adj[i].append(j)
    ans = 0
    #DFS로 매칭점을 찾는다
    for i in range(n) :
        visited = [False for j in range(n)]
        flag = dfs(i)
        if flag == True :
            ans = ans+1

    #경로찾기, BFS를 사용한다. match가 -1이면 쓰이지 않은 라인 = bfs
    visited = [False for j in range(n)]
    for i in range(n) :
        if visited[i] == False and match[i] == -1 :
            bfs(i)

    #커버 하기 위해 지워진 행
    Except_X = []
    #커버 하기 위해 지워진 열
    Except_Y = []
    for i in range(n) :
        if CoverA[i] == True :
            Except_X.append(i)
        if CoverB[i] == True :
            Except_Y.append(i)
    return Except_X,Except_Y,ans

# 헝가리안 알고리즘
def Hungarian(Map, n) :

    #모든 행에 대해서, 그 행의 각 원소에 그 행에서 가장 작은 값을 뺀다.
    for i in range(0,n) :
        min_len = 99999999
        for j in range(0,n) :
            min_len = min(Map[i][j], min_len)
        for j in range(0,n) :
            Map[i][j] = Map[i][j] - min_len

    #모든 열에 대해서, 그 열의 각 원소에 그 열에서 가장 작은 값을 뺀다.
    for i in range(0,n) :
        min_len = 99999999
        for j in range(0,n) :
            min_len = min(Map[j][i], min_len)
        for j in range(0,n) :
            Map[j][i] = Map[j][i] - min_len

    #행과 열을 n개보다 적게 뽑아서, 행렬의 모든 0의 값을 갖는 원소를 덮는 방법이 없을 때 까지 아래를 반복한다.
    while True :
        Map2 = copy.deepcopy(Map) #기존의 맵은 konig  알고리즘 수행중에는 손항되어선 안됩니다.
        Except_X,Except_Y,cnt = Konig(Map2,n) #konig알고리즘을 통해 커버할 열과 행을 정합니다
        if cnt == n :
            break
        #뽑힌 곳을 제외하고 가장 작은 수를 구합니다
        min_len = 99999999
        for i in range(n) :
            for j in range(n) :
                if Except_X.count(i) == 0 and Except_Y.count(j) == 0 :
                    min_len = min(min_len, Map[i][j])
        # Except_X 에 속하지 않는 행에 대해서만 최소 비용 뺄샘을 진행합니다
        for i in range(n) :
            if i not in Except_X :
                for j in range(n) :
                    Map[i][j] = Map[i][j] - min_len
        # Except_Y에 속하는 열에 대해서 최소 비용 덧샘을 진행합니다.
        for i in Except_Y :
            for j in range(n) :
                Map[j][i] = Map[j][i] + min_len
        
    #과정이 끝나면 DFS로 배치를 시작합니다.
    # deq의 원소는 현재까지 매칭리스트 visit과, 현재 행으로 이루어집니다.
    visit = [-1 for i in range(n)]
    deq = col.deque()
    for i in range(n) :
        if Map[0][i] == 0 :
            ivisit = copy.deepcopy(visit)
            ivisit[i] = 0
            deq.append([1, ivisit])
    while len(deq) != 0 :
        x,visit = deq.pop()
        if x == n :
            break
        for i in range(n) :
            if visit[i] == -1 and Map[x][i] == 0 :
                ivisit = copy.deepcopy(visit)
                ivisit[i] = x
                deq.append([x+1, ivisit])

    #visit에 매칭 리스트가 적혀저있습니다. (번지 수 : 열 - 번지 값 : 행)
    result = []
    for i in range(n) :
        result.append([visit[i], i])
    return result
        

#예측 지점과 측정 지점 간의 거리를 반환한다. 행 : 예측점 열 : 측정점
def makedistance(Predict_list, Measured_list) :
    p_len = len(Predict_list)
    m_len = len(Measured_list)
    n = max(p_len,m_len)
    # 예측점과 측정점 중 더 개수가 많은 값을 중심으로 정방 행렬을 만들어야하며, 길이가 다른 경우 가상의 거리를 만들어서 사용한다.
    Distance_list = [[9999999 for i in range(n)] for j in range(n)] 

    for i in range(0,p_len) :
        for j in range(0,m_len) :
            x_dis = Predict_list[i][0] - Measured_list[j][0]
            y_dis = Predict_list[i][1] - Measured_list[j][1]
            Distance_list[i][j] = math.sqrt(math.pow(x_dis,2) + math.pow(y_dis,2))
    return Distance_list


def calculator(dogs , measured, gridbox, interval) :

    #강아지의 번호를 위한 배열
    num_visit = [False for i in range(10)]

    #현재 존재중인 강이지들의 예측지점을 칼만 필터를 통해 생성한다.
    predict= []
    for now_dog in dogs :
        predict.append(now_dog.KalmanFilter_Predict(interval))
    
    #예측지점과 측정 지점 간의 거리를 측정한다.
    Distance_list = makedistance(predict, measured)
    # 헝가리안 알고리즘을 통해 예측지점과 측정지점을 매칭 시킵니다. matching의 각 요소는  0번지가 예측점, 1번지가 측점지점로 구성됩니다.
    matching = Hungarian(Distance_list, n)

    #삭제되어야 할 객체와 새로 생성되어야할 객체를 판단합니다.
    del_list = []
    new_list = []

    # 만약에 가상예측지점과 연결된 측정지점이있다면 객체를 새로 생성해야할 것입니다.
    # 반면, 가상측정지점과 연결된 예측지점이 있다면 객체를 해당 객체를 삭제해야 할 것입니다.
    # 두 경우 모두 아닐 경우 칼만필터를 비롯한 상태값을 갱신합니다.
    for match in matching :
        if match[0] >= len(predict) :
            new_list.append(match[1])
        elif match[1] >= len(measured) :
            del_list.append(match[0])
        else :
            dogs[match[0]].KalmanFilter_Correct(measured[match[1]])
            dogs[match[0]].newflag = False 
            dogs[match[0]].updater(measured, match, interval, gridbox)
            num_visit[dogs[match[0]].number] = True

    #소멸
    if len(del_list)!= 0 :
        imsi_dogs = []
        for i in range(len(dogs)) :
            if i not in del_list :
                imsi_dogs.append(dogs[i])
            else :
                del dogs[i]
        dogs = imsi_dogs #안되면 카피로 할 것

    #생성
    elif len(new_list)!=0 :
        for i in range(len(measured)) :
            if i in new_list :
                for j in range(1,10) :
                    if num_visit[j] == False :
                        dogs.append(dogs(measured[i][0],measured[i][1],gridbox[i],j))
                        num_visit[j] = True
                        break
    
    angles = dict()
    for dog in dogs :
        angles.update({dog.number : Define_Angle(dog.ymax,dog.xmin,dog.max)})
    return angles,dogs



