import math
import copy
import collections as col
import numpy as np


class dog :
    #새로 생성할 것이 필요할 경우 생성한다.

    def __init__(self,x,y) :
        self.x = x
        self.y = y
        self.x_v = 0 #벡터값
        self.y_v = 0 #벡터값

        self.label =""
        self.ymin = 0
        self.xmin = 0
        self.xmax = 0
        self.ymax = 0

        self.newflag = True

        self.Kalman_Gain = np.array([[0,0,0,0],[0,0,0,0] , [0,0,0,100] , [0,0,100,0] ]) #초기값 배열
        self.Kalman_P_now = np.array([[0,0,0,0],[0,0,0,0] , [0,0,0,100] , [0,0,100,0] ]) #초기값 배열     
        self.Kalman_P_predict = np.array([[0,0,0,0],[0,0,0,0] , [0,0,0,100] , [0,0,100,0] ])
        self.Kalman_z = np.array([[self.x],[self.y]])
        self.Kalman_x_now = np.array([[self.x],[self.x_v],[self.y],[self.y_v]])
        self.Kalman_x_predict = np.array([[self.x],[self.x_v],[self.y],[self.y_v]])

    #예측 값을 보낸다.
    def KalmanFilter_Predict(self, interval) :
        A=np.array([[1, interval ,0,0],[0,1,0,0] , [0,0,1, interval] , [0,0,0,1] ]) #상태벡터 도출 벡터 0.2가 시간인데 시간 값이 바뀔 필요가 있음.
        Q=np.array([[1,0,0,0],[0,1,0,0] , [0,0,1,0] , [0,0,0,1] ])

        self.Kalman_x_predict= A @ self.Kalman_x_now
        self.Kalman_P_predict = A @ self.Kalman_P_now @ np.transpose(A) + Q
        return [self.Kalman_x_predict[0][0],self.Kalman_x_predict[2][0]] #0,2아닌가 제대로는 다시 봐야할거같음!! 이슈이슈

    
    #맞는 xy값을 줘서 칼만 필터를 수정한다
    def KalmanFilter_Correct(self,correct) :
        correct_z_x = correct[0]
        correct_z_y = correct[1]

        H=np.array([[1,0,0,0],[0,1,0,0]])
        R=np.array([[50,0],[0,50]])
        I=np.array([[1,0,0,0],[0,1,0,0] , [0,0,1,0] , [0,0,0,1] ])
        self.Kalman_z = np.array([[correct_z_x],[correct_z_y]])
        self.Kalman_Gain =( self.Kalman_P_predict @ np.transpose(H) ) @ np.linalg.inv(H @ self.Kalman_P_predict @ np.transpose(H) + R )
        self.Kalman_x_now = self.Kalman_x_predict + self.Kalman_Gain @ (self.Kalman_z - H @ self.Kalman_x_predict)
        self.Kalman_P_now = (I- self.Kalman_Gain @ H ) @ self.Kalman_P_predict

    #현재 각도 를 반환한다
    def set_angle(self) :

        angle = 0
        if self.y < 0 :
            return 0


        if self.x < 0 :
            angle  = math.atan2((-1 *self.x) , self.y)* (180/3.14) * -1
        else :
            angle  = math.atan2(self.x , self.y)* (180/3.14)
        return angle


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

#쾨니그는 한번 더 이해하기 위해 노력할 필요가 있음. 기본적으로 이분매칭 방법을 사용함
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

    #매칭점 찾기! = dfs
    for i in range(n) :
        visited = [False for j in range(n)]
        flag = dfs(i)
        if flag == True :
            ans = ans+1

    #경로찾기! match가 -1이면 쓰이지 않은 라인 = bfs
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




def Hungarian(Map, n) :
    # 헝가리안 알고리즘 시작
    
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
        min_len = 99999999
        for i in range(n) :
            for j in range(n) :
                #뽑힌 곳을 제외하고 가장 작은 수를 구합니다
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
        now = deq.pop()
        x = now[0]
        visit = now[1]
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
        


def makedistance(Predict_list, Measured_list) :
    p_len = len(Predict_list)
    m_len = len(Measured_list)
    n = max(p_len,m_len)
    # 예측점과 측정점 중 더 개수가 많은 값을 중심으로 정방 행렬을 만들어야하며,
    # 길이가 다른 경우 가상의 거리를 만들어서  사용한다.
    Distance_list = [[9999999 for i in range(n)] for j in range(n)] 


    #각 지점간의 거리를 측정한다.

    #행 : 예측점 열 : 측정점
    for i in range(0,p_len) :
        for j in range(0,m_len) :
            x_dis = Predict_list[i][0] - Measured_list[j][0]
            y_dis = Predict_list[i][1] - Measured_list[j][1]
            Distance_list[i][j] = math.sqrt(math.pow(x_dis,2) + math.pow(y_dis,2))
            
    result = Hungarian(Distance_list, n)
    # 예측점 - 측정점간 매칭으로 이루어진 값을 보냅니다.
    return result 





def calculator(dogs , measured, gridbox, interval) :
    


    predict= []

    for now_dog in dogs :
        predict.append(now_dog.KalmanFilter_Predict(interval))
    
    #바운딩 박스 처리하기 전에 mesaured를 채우겠지? change real parameter를 통해서


    matching = makedistance(predict, measured)

    
    # match 는 0번지가 예측점, 1번지가 측점지점
    del_list = []
    new_list = []
    for match in matching :
        #만약에 가상예측지점과 연결된 측정지점이있다면 측정지점을 새로 생성해야할것
        if match[0] >= len(predict) :
            new_list.append(match[1])
        #만약에 가상측정지점과 연결된 측정지점이있다면 측정지점을 새로 생성해야할것
        elif match[1] >= len(measured) :
            del_list.append(match[0])
        else :
            #정상적인 매칭의 경우, 칼만 필터를 수정합니다. 또한 매칭이 되었을 경우 새로운 친구는 아니라 할 수 있음으로 newflag를 수정합니다.
            dogs[match[0]].KalmanFilter_Correct(measured[match[1]])
            dogs[match[0]].newflag = False


            #또한 위치와 속도도 바꿔야 할 것이다.
            dogs[match[0]].x_v = (measured[match[1]][0] - dogs[match[0]].x) / interval
            dogs[match[0]].y_v = (measured[match[1]][1] - dogs[match[0]].y) / interval
            dogs[match[0]].x = measured[match[1]][0]
            dogs[match[0]].y = measured[match[1]][1]
            dogs[match[0]].ymin = gridbox[match[1]][0]
            dogs[match[0]].xmin = gridbox[match[1]][1]
            dogs[match[0]].ymax = gridbox[match[1]][2]
            dogs[match[0]].xmax = gridbox[match[1]][3]

    
    #예측 지점이 측정 지점보다 많을 경우 소멸이 되어야 합니다. del 로 소멸하는 것도 추가하자.
    if len(del_list)!= 0 :
        imsi_dogs = []
        for i in range(len(dogs)) :
            if i not in del_list :
                imsi_dogs.append(dogs[i])
        dogs = imsi_dogs #안되면 카피로 할 것

    
    #측정 지점이 예측 지점보다 많을 경우 생성이 되어야 합니다.
    elif len(new_list)!=0 :
        for i in range(len(measured)) :
            if i in new_list :
                dogs.append(dog(measured[i][0],measured[i][1]))
                dogs[len(dogs)-1].ymin = gridbox[i][0]
                dogs[len(dogs)-1].xmin = gridbox[i][1]
                dogs[len(dogs)-1].ymax = gridbox[i][2]
                dogs[len(dogs)-1].xmin = gridbox[i][3]
    
    
    
    if len(dogs) != 0 :
        rotate_angle = 99999999
        p_x = 99999999
        p_y =9999999

        #확인용 cnt
        p_cnt = 1 
        for a_dog in dogs :
            n_angle = a_dog.set_angle()
            print("dog" , p_cnt , " angle: " ,n_angle )
            #가장 작은 각도에 존재하는 강아지를 위주로 회전할 수 있도록 한다. 그리고, 맨 처음 검출된 녀석의 경우 오검출의 가능성이 있기 때문에 추적에서 제외한다.
            if abs(rotate_angle) > abs(n_angle) and a_dog.newflag ==False :
                rotate_angle = n_angle
                p_x = a_dog.x
                p_y = a_dog.y
        print("rotate_angle :" , rotate_angle,"x :" ,p_x, "y :", p_y)
        if rotate_angle == 99999999 :
            rotate_angle = 0
        return rotate_angle,dogs
    else :
        return 0,dogs

    




