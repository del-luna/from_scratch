#Backpropagation!
# x -> [A] -> a -> [B] -> b -> [C] -> y
#dy/dx = dy/db*db/da*da/dx
#dy/dx = ((dy/dy*dy/db)*db/da)*da/dx  (dy/dy = 1)
#dy/dy, dy/db, dy/da, dy/dx is derivative of y with respect to (y, b, a, x)

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    '''
    Base class
    specific functions are implemented in the inherited class
    '''
    def __call__(self, input): 
        x = input.data #data extract
        y = self.foward(x)
        output = Variable(y)
        self.input = input #keep variable for use in backward()
        return output

    def foward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
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



A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)