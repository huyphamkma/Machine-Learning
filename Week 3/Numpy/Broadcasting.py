import numpy as np

'''
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # tao mang rong y co cung kich thuoc vs mang x


for i in range(4):
    y[i, :] = x[i, :] + v   # cong vector v vao moi hang cua y

# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
'''

'''
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # xep chong 4 ban sao cua v len nhau
print(vv)                 #        "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # cong x va vv
print(y) 	# [[ 2  2  4]
			#  [ 5  5  7]
			#  [ 8  8 10]
			#  [11 11 13]]
'''


x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # cong v vao moi hang cua x su dung broadcasting
print(y)  #		    [[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]