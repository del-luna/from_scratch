import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        '''
        1. 이 변수가 어디서 왔는지를 가져옴(creator)
        2. creator의 인풋을 가져옴
        3. creator의 backward method에 이전 gradient를 집어넣고 gradient 계산
        4. 재귀적으로 다음 스텝으로 넘어감
        '''
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

class Function:
    def __call__(self, input):
        '''
        1. 데이터를 받음
        2. foward를 통해 연산
        3. 연산 결과를 Variable 클래스를 통해 저장(이게 키포인트)
        4. 연산 결과를 담고있는 Variable에 creator라는 어떤 함수의 결과인지 네임 태그를 달아줌
        5. input(backward 연산 때 쓰려고) & ouput(이건 왜 저장하지? -> 일단 지금은 필요 없음)을 저장함.
        '''
        x = input.data
        y = self.foward(x)
        output = Variable(y) #memory!!! keypoint
        output.set_creator(self)
        self.input = input
        #self.output = output
        return output

    def foward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

class Square(Function):
    def foward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def foward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

x = Variable(np.array(0.5))
A = Square()
B = Exp()
C = Square()

a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)