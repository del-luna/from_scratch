#implement variable!
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
        output = Variable(y)
        return output

    def foward(self, x):
        raise NotImplementedError()

class Square(Function):
    def foward(self, x):
        return x ** 2


#Example.2
x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)


