# BA_02 Kernel-based Learning: Support Vector Machines (SVM)

<p align="center"><img width="467" alt="image" src="https://user-images.githubusercontent.com/97882448/199305047-75333530-7b59-4745-829b-c029a3cfbef5.png">

 ## BA_02 Support Vector Machines (SVM)_개념설명
  
이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Support Vector Machines을 바탕으로 만들어졌습니다.
먼저, Support Vector Machines의 기본적인 개념에대해서 설명하고 Jupyter notebook을 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.
또한, 여러가지 파라미터들을 변경해보면서 분류경계면이 바뀌는 것또한 확인해보겠습니다. 
설명할때 통일성을 위해 Support Vector Machines을 SVM라고 명명하도록하겠습니다.

SVM은 쉽게 이야기하면 데이터를 분류하는 최적의 선을 찾는것입니다. 조금 더 자세히 말하자면 초평면을 생성해 마진을 최대화하는 분류경계를 찾는 것입니다. 이말을 들으시면 SVM을 처음 공부하시는 분들은 아래의 말하는 감자그림처럼 초평면?, 마진? 이것이 무엇인지 궁금해지실겁니다. 
 <p align="center"><img width="250" alt="image" src="https://user-images.githubusercontent.com/97882448/199414674-169d3bd0-b51e-48b8-8cca-73dbcb7fa590.png"> 
   
그래서 초평면과 마진이 무엇인지 쉽게 설명해보겠습니다.
 <p align="center"><img width="350" alt="image" src="https://user-images.githubusercontent.com/97882448/199418846-104a9b8a-a3c9-4385-86e8-865a1cbb6c83.png">

* 초평면: 위의 그림처럼 그룹을 나누는 이상적인 평면(=직선)이라고 생각하시면 되겠습니다. 2차원일땐 직선으로 분류되고 3차원일땐 평면으로 분리가 됩니다. 
   
 <p align="center"><img width="250" alt="image" src="https://user-images.githubusercontent.com/97882448/199420189-93969137-a9e9-453a-a5b1-655157e97c84.png">

* 마진: 위의 그림처럼 그룹을 나누는 직선에서 수직으로 벡터를 그렸을때 처음만나는 점과의 거리입니다. 위의그림처럼 마진은 빨간점과 파란점의 거리라고도 하는데 다른교재에선 초평면에서 점과의 거리라고 부르기도 합니다. 

초반에 제가 최적의 선을 찾아야한다고 했는데 눈치가 빠르신분이면 생각하셨을겁니다. 혹시 최적이면 기준이 있어야 하지 않을까? 과연 기준이 무엇일까? 라고요.
제가 퀴즈를 내볼테니 한번 찾아보시면 감사하겠습니다. 제가 Jupyter Notebook을 이용해 한번 만들어보았습니다.(만드는 방법은 코드설명란에서 자세히 다루겠습니다.)
   
 <p align="center"><img width="350" alt="image" src="https://user-images.githubusercontent.com/97882448/199422766-f554c476-c2ce-4542-8ca4-aef62e2366dd.png">


