import numpy as np
import unittest

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #y.grad = np.array(1.0) 

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    '''
    Base class
    specific functions are implemented in the inherited class
    '''
    def __call__(self, input): 
        x = input.data #data extract
        y = self.foward(x)
        output = Variable(as_array(y))
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

def square(x):
    return Square()(x)

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

class SquareTest(unittest.TestCase):
    def test_foward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad) #얼마나 가까운지 확인
        self.assertTrue(flg)

unittest.main()