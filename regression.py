import numpy
import matplotlib.pyplot as plt
import math

# creating the equation Xb = y

# X = [1 | x | x^2]
X = numpy.array((
    [1, 0, 0],
    [1, 1, 1],
    [1, 2, 4],
    [1, 3, 9],
    [1, 4, 16],
    [1, 5, 25],
    [1, 6, 36],
    [1, 7, 49],
    [1, 8, 64],
    [1, 9, 81],
    [1, 10, 100],
    [1, 11, 121],
    [1, 12, 144],
))

# Y = position of airplane
y = numpy.array((
    [0],
    [5],
    [18],
    [63],
    [104],
    [159],
    [222],
    [277],
    [350],
    [481],
    [570],
    [677],
    [832],
))



# creating the equation (XtX)b = (Xt)y
XtX = numpy.linalg.multi_dot((numpy.transpose(X), X))
Xty = numpy.linalg.multi_dot((numpy.transpose(X), y))

# solving the equation
coefficients = numpy.linalg.solve(XtX, Xty)

predicted = []
for i in range(len(y)):
    predicted.append(math.floor(numpy.linalg.multi_dot((X[i], coefficients))))


plt.xlabel('Time (seconds)')
plt.ylabel('Altitude of Airplane')
plt.plot(y, 'ro')
plt.plot(predicted)
plt.show()
