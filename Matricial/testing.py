import numpy as np

m1 = np.matrix([[1,2,0],[0,1,2]])
m2 = np.matrix([1,4,3]).T
m3 = m1 @ m2
m4 = np.matrix([1,1]).T
m5 = m3 + m4
m6 = np.matrix([2,3]).T
m7 = np.multiply(m5,m6)
m8 = np.multiply(m1,m6)
m9 = m8 * 5
m10 = m9.sum(0)
m11 = m10 - 1
m12 = np.matrix(np.zeros((2,3)))
m13 = m2.A1
m14 = m12.A1
m15 = np.matrix(np.random.rand(3, 4) * 2 - 1)
m16 = m15.A1

print(m1)
print(m2)
print(m3)
print(m4)
print(m5)
print(m6)
print(m7)
print(m8)
print(m9)
print(m10)
print(m11)
print(m12)
print(m13)
print(m14)
print(m15)
print(m16)
