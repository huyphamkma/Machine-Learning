import numpy as np

'''
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)


print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))
'''
'''
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))

print(x.dot(v))
print(np.dot(x, v))

print(x.dot(y))
print(np.dot(x, y))
'''

'''
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # tong cac phan tu
print(np.sum(x, axis=0))  # tong tung cot
print(np.sum(x, axis=1))	# tong tung hang
'''



x = np.array([[1,2], [3,4]])
print(x)    
            
print(x.T)  # hoan vi mang x
            


v = np.array([1,2,3])
print(v)    
print(v.T)