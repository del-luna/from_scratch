#AutoGrad

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator #1. Bring function
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward() #recursive

class Function:
    '''
    Base class
    specific functions are implemented in the inherited class
    '''
    def __call__(self, input): 
        x = input.data #data extract
        y = self.foward(x)
        output = Variable(y)
        output.set_creator(self) # creator records every time it is calculated
        self.input = input #keep input for use in backward()
        self.output = output #keep output
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
'''
#Code before implementing the backward method of variable
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#Define-by-Run
#Connections are 'defined' when flowing data from the foward('Run')
assert y.creator == C 
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)

C = y.creator #1. Bring function
b = C.input #2. Bring input of the function
b.grad = C.backward(y.grad) #3. call backward()

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)
'''

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#Backpropagation
y.grad = np.array(1.0)
y.backward()
print(x.grad)