#Implement numerical differentiation
#Approximiation of 'real' differentiation using h = 1e-4 (small value)
#Use the central difference to reduce the error

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    '''
    Base class
    specific functions are implemented in the inherited class
    '''
    def __call__(self, input): 
        x = input.data #data extract
        y = self.foward(x)
        output = Variable(y) #here! is key point
        return output

    def foward(self, x):
        raise NotImplementedError()

class Square(Function):
    def foward(self, x):
        return x ** 2

class Exp(Function):
    def foward(self, x):
        return np.exp(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps) #x-h
    x1 = Variable(x.data + eps) #x+h
    y0 = f(x0) #f(x-h)
    y1 = f(x1) #f(x+h)
    return (y1.data - y0.data) / (2 * eps) # f(x+h)-f(x-h)/2h

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

'''
# numerical diff
f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)
'''

# composite function differentiation
x = Variable(np.array(0.5))
dy = numerical_diff(f, x) # np.exp(0.5**2)*np.exp(0.5**2)*2 == 2exp(0.5^2) * exp(0.5^2)
print(dy) # Changing x by a small value(h=1e-4) changes y by a factor of 3.297...