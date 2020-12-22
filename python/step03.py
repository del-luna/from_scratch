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

# input/output of a Function.__call__  is unified as a variable instance.
square = Square()
exp = Exp()

# like a composite function
# x -> [Square] -> a -> [Exp] -> b -> [Square] -> y
x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)
print(y.data)


