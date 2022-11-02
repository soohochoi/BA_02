# BA_02 Kernel-based Learning: Support Vector Machines (SVM)

<p align="center"><img width="467" alt="image" src="https://user-images.githubusercontent.com/97882448/199305047-75333530-7b59-4745-829b-c029a3cfbef5.png">

 ## BA_02 Support Vector Machines (SVM)_개념설명
  
이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Support Vector Machines을 바탕으로 만들어졌습니다.
먼저, Support Vector Machines의 기본적인 개념에대해서 설명하고 Jupyter notebook을 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다.
또한, 여러가지 파라미터들을 변경해보면서 분류경계면이 바뀌는 것또한 확인해보겠습니다. 
설명할때 통일성을 위해 Support Vector Machines을 SVM라고 명명하도록하겠습니다.

SVM은 쉽게 이야기하면 선형 분류기(=선형으로 분리됨)이며 데이터를 분류하는 최적의 선을 찾는것입니다. 조금 더 자세히 말하자면 초평면을 생성해 마진을 최대화하는 분류경계를 찾는 것입니다. 이말을 들으시면 SVM을 처음 공부하시는 분들은 아래의 말하는 감자그림처럼 초평면?, 마진? 이것이 무엇인지 궁금해지실겁니다. 
 <p align="center"><img width="250" alt="image" src="https://user-images.githubusercontent.com/97882448/199414674-169d3bd0-b51e-48b8-8cca-73dbcb7fa590.png"> 
   
그래서 초평면과 마진이 무엇인지 쉽게 설명해보겠습니다.
 <p align="center"><img width="350" alt="image" src="https://user-images.githubusercontent.com/97882448/199418846-104a9b8a-a3c9-4385-86e8-865a1cbb6c83.png">

* 초평면: 위의 그림처럼 그룹을 나누는 이상적인 평면(=직선)이라고 생각하시면 되겠습니다. 2차원일땐 직선으로 분류되고 3차원일땐 평면으로 분리가 됩니다. 
   
 <p align="center"><img width="250" alt="image" src="https://user-images.githubusercontent.com/97882448/199420189-93969137-a9e9-453a-a5b1-655157e97c84.png">

* 마진: 위의 그림처럼 그룹을 나누는 직선에서 수직으로 벡터를 그렸을때 처음만나는 점과의 거리입니다. 위의그림처럼 마진은 빨간점과 파란점의 거리라고도 하는데 다른교재에선 초평면에서 점과의 거리라고 부르기도 합니다. 

초반에 제가 최적의 선을 찾아야한다고 했는데 눈치가 빠르신분이면 생각하셨을겁니다. 혹시 최적이면 기준이 있어야 하지 않을까? 과연 기준이 무엇일까? 라고요.
제가 퀴즈를 내볼테니 한번 찾아보시면 감사하겠습니다. 제가 Jupyter Notebook을 이용해 한번 만들어보았습니다.    
   **(만드는 방법은 코드설명란에서 자세히 다루겠습니다.)**
   
 <p align="center"><img width="350" alt="image" src="https://user-images.githubusercontent.com/97882448/199422766-f554c476-c2ce-4542-8ca4-aef62e2366dd.png">
  
그림을 참조하시면 파란선과 연두선과 주황선이 보이실겁니다. 빨간공과 보라색공을 나름대로 다 분류를 잘한것 같은데 어떤것이 제일 잘한것처럼 보이시나요? 아까 설명드렸던 
마진을 기억해내시면 아래에 그림처럼 연두색이 가장 좋은 분류경계선을 만들었다! 라고 생각하실겁니다! 왜냐하면 연두색의 마진이 가장크기 때문입니다.
  
 <p align="center"><img width="350" alt="image" src="https://user-images.githubusercontent.com/97882448/199424779-29e2ed32-57d1-47cd-b68c-8724c5ef8921.png">

그럼 좀더 나아가 SVM에서 Hard마진과 Soft마진에 대해서 공부해봅시다. 이름만 들어도 Hard는 무언가? 딱딱하고 깐깐할것 같고 Soft는 유연하고 부드러울것 같다고 생각하신 분이 있으신가요? 100% 정답이라고는 말하지 못하지만 감이 좋다고 말할수 있습니다.아래의 그림으로 이해하기 쉽게 설명해보겠습니다.
 <p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/199426782-e5d5ebff-b06c-44fd-826a-204e4d5f8a51.png">

### * Hard마진

 <p align="center"><img width="500" alt="스크린샷 2022-11-02 오후 4 50 48" src="https://user-images.githubusercontent.com/97882448/199430738-91234162-0163-43c4-90b1-737d81ddb0cd.png">
  
  
1. 그림을 보신것처럼 마진의 값을 타이트하게 잡아서 한개의 이상치조차 용납하지 않습니다. 
2. 때문에 과적합이 발생하기 쉽고 노이즈로 인해 최적의 경계면을 잘못구하거나 못찾는경우가 발생할수도 있습니다.

### * Soft마진

 <p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/199431479-4911bac4-1338-4d3e-8e67-5c491d503f11.png">


1. 그림을 보신것처럼 Hard마진의 한계를 극복하기위해 분류경계면 안에 있는 아이들을 인정하는 대신 패널티(크사이)를 주는 방식이라고 생각하면 됩니다.
2. 이상치를 어느정도 허용하면서 결정경계를 설정하는 방법입니다.

위 식들을 푸는 방법은 제약식에 라그랑지안 승수를 곱함 -> 라그랑지안 Primal 문제로 바꿈 -> KKT condition을 통해 Dual문제로 바꿈 -> Solution을 푸는 방법은 tutorial보다 ppt설명에 넣는것이 맞는것 같아 이번시간 내용으론 제외 시켰습니다. 

그럼 Hard마진도 알겠고 Soft마진도 알겠는데 SVM은 선형모델만 분류할수있는건가요? 물을수 있습니다. 현실에선 선형보다 비선형도 분류하는 밀도있고 복잡한 데이터가 많기때문입니다.    
제가 자신있게 자문자답을 한 이유는 SVM은 비선형분류기도 가능하기 때문입니다. 그러면 어떻게 하면 될까요? 답은 Kernal trick을 통해서입니다. Kernal trick이란 원래공간이 아닌 선형분류가 가능한 더 고차원의 공간으로 데이터를 보내서 경계면을 설정하는 방법입니다. 아래의 그림을 참고하면 이해하는데 좀더 용이할것입니다.
  
   <p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/97882448/199435448-b67265bc-c8e6-4678-a114-5c17843e1c36.png">

그럼 지금까지 배웠던 내용을 가지고 실습을 해보는 시간을 가지도록하겠습니다. 
    
 ## BA_02 Support Vector Machines (SVM)_실습코드 

먼저 SVM 그림을 만들어보겠습니다. 

```python
#package를 통해 불러옴
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# use seaborn plotting defaults
import seaborn as sns; sns.set()
```
* 먼저 SVM 그림을 만들때 사용한 패키지입니다. 
    
```python
#점을 생성하는 package를 불러옴
from sklearn.datasets import make_blobs
# 점을 생성하는  sample의 갯수,centers는 종류, cluster_std는 점들의 분산 정도임
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=4, cluster_std=0.4)
#x,y축 그래프를 만드는데 s는 점의 크기를 결정하고 c=y를 통해 2개의 색깔을 fix 시킴
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='rainbow')
```
* 점을 랜덤으로 생성하는 package를 통해서 sample의 갯수, 종류, 점들의 분산을 설정함 
* scatter을 통해 x축,y축을 설정하고 s를 통해 원의 크기를 설정함
    
   <p align="center"><img width="376" alt="image" src="https://user-images.githubusercontent.com/97882448/199447796-c6710161-d63e-4b12-a561-9956adb405f7.png">
  
```python
 xfit = np.linspace(7.8, 11)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='rainbow')
#보라색과 빨간색 사이를 마구잡이로 경계를 그었음
for m, b in [(-0.5, 7.9), (0.53, -2.8), (-0.09, 3.4)]:
    plt.plot(xfit, m * xfit + b, '--')

plt.xlim(7.8, 11);
```
