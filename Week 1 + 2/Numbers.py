number1 = 0xA0F
print(number1)
number2 = 0o37
print(number2)

print("abs(-45) : ", abs(-45))
print("abs(100.12) : ", abs(100.12))

import math
#ceil(x) : so nguyen gan nhat lon hon x
print("math.ceil(-45.17) : ", math.ceil(-45.17))
print("math.ceil(45.111) : ",math.ceil(45.111))

print("max(10,1,20) : ", max(10,1,20))

print("math.pow(100,2): ",math.pow(100,2))

print("math.sqrt(100) : ", math.sqrt(100))

import random

print ("returns a random number from range(100) : ",random.choice(range(100)))
print ("returns random element from list [1, 2, 3, 5, 9]) : ", random.choice([1, 2, 3, 5, 9]))
print ("returns random character from string 'Hello World' : ", random.choice('Hello World'))

# random so le tu 1 -> 100
print ("randrange(1,100, 2) : ", random.randrange(1, 100, 2))

# random 1 so tu 1 -> 99
print ("randrange(100) : ", random.randrange(100))

# Random 1 so tu [0.0, 1.0]
print ("random() : ", random.random())

# Random vị trí các phần tử trong list
list = [20, 16, 10, 5];
random.shuffle(list)
print ("Reshuffled list : ",  list)