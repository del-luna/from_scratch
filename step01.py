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

'''
#Example.1
data = np.array(1.0)
x = Variable(data) # Variable instance, x is not data, just box!
print(x)

x.data = np.array(2.0) #input new data
print(x)
'''

x = Variable(np.array(10))
f = Function()
y = f(x)
print(type(y))
print(y.data)