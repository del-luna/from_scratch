# -*- coding: utf-8 -*-
import weakref
import contextlib
import numpy as np

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

class Config:
    enable_backprop = True

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0 # generation record

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)
    
    def cleargrad(self):
        self.grad = None

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
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
            gys = [output().grad for output in f.outputs] # 약한 참조로 받아서 값 보려면 ()추가
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
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

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

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs #keep input for use in backward()
            self.outputs = [weakref.ref(output) for output in outputs] #순환참조 끊기
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

class Mul(Function):
    def foward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    return Mul()(x0, x1)

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)