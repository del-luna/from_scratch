import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # generation record

    def cleargrad(self):
        self.grad = None

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #y.grad = np.array(1.0) 

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            '''
            여기서 loop 돌면서 f.backward 호출
            f = Add, gys = 1.
            gxs = (1., 1.) : Add에 대한 gradient는 1
            f = Square, Sqaure, gys = 1.
            이제 각각 Sqaure에 대한 input이 다르므로 각각의 gradient 계산.
            '''
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

class Function:
    '''
    Base class
    specific functions are implemented in the inherited class
    '''
    def __call__(self, *inputs): 
        xs = [x.data for x in inputs]
        ys = self.foward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,) 
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs #keep input for use in backward()
        self.outputs = outputs 
        return outputs if len(outputs) > 1 else outputs[0]

    def foward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def foward(self, x0, x1):
        y = x0 + x1
        return (y,)
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def foward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

class Exp(Function):
    def foward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps) #x-h
    x1 = Variable(x.data + eps) #x+h
    y0 = f(x0) #f(x-h)
    y1 = f(x1) #f(x+h)
    return (y1.data - y0.data) / (2 * eps) # f(x+h)-f(x-h)/2h
'''
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)
'''
import time
t1 = time.time()
for _ in range(10):
    x = Variable(np.random.randn(100000))
    y = square(square(square(x)))
print(time.time()-t1)