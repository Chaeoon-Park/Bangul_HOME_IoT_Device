


var HomeDB = {
    'num_of_dogs' : 0,
    
    're_detect' : false, // true일 경우 재탐색을 실행합니다.

    'trace_dog' : 0, //0일경우에는 자동으로 현재 기준 가장 가까운 강아지를 따라다닙니다. 번호가 지정될 경우 지정된 번호의 강아지를 쫒아다니게 됩니다.

    'auto_mode' : true, // false가 되면 각도 관련 함수들을 실행 시킵니다.

    'rotate_direction' : 'none',

    'set_middle' : false,

    'home_running' : true
}



// # 1. 강아지가 현재 몇마리 검출 중인가 ->num_of_dog

// # 서버에서 내가 받고 싶은 것

// # 1. 강아지가 없었을 때 강아지 재탐색을 실행 할 것인가

// # 1-1) 재탐색 실행 명령어 -> re_detect = true
// # 1-2) 재탐색시에도 발견 되지 않으면 어떻게 멈출 것인가 -> re_detect =false
// # 1-3) 재탐색 신호가 들어오지 않고(re_detect = false) 오랫동안 request를 줘야하는 상황이면 어떻게 할 것인가? (쓸데없이 트래픽을 많이 주게 됨) -> re_detect = true 가 올 때 까지는 리퀘스트 주기를 천천히함


// # 2. 강아지가 있을 때 

// # 2-1) 추적하고 싶은 강아지가 있다면 그 강아지를 쫒을것인가 ->  trace_dog = number of dog
// # 2-2) 추적하고 싶은 강아지가 없다면 어떤 로직으로 각도를 설정 할 것인가. -> trace_dog = 0 일 경우  


// # 3. 수동 조작을 원할 때

// # auto_mode = true : 자동 auto_mode : false 수동

// # auto_mode = false일때만 받아오는 함수들
// # rotate_direction : 'right' 일때는  오른쪽으로, 'left'일때는 왼쪽으로 10도 움직임. 'none' 일때는 움직임 없음. 이것 역시 명령을 내린 뒤 'none'으로 항상 바꿔야할 것.
// # set_middle : true 일경우 90도 지점으로 움직임 바로 set_middle 은 명령을 내린 뒤 false로 바뀌어야할것


// # 4. 디바이스 자체를 멈추고 싶을 때

// #home_runnig이 false가 되면 sleep 모드로 들어가고 주기적으로 리퀘스트만 보낼 수 있도록 한다.