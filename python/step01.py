#implement variable!
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

#Example.1
data = np.array(1.0)
x = Variable(data) # Variable instance, x is not data, just box!
print(x.data)

x.data = np.array(2.0) #input new data
print(x.data)


