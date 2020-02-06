list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1,2,3,4,5,6,7]

print("list1[0]: ", list1[0])
print("list2[1:5]: ", list2[1:5])

#update list
list = ['physics', 'chemistry', 1997, 2000]
print ("Value available at index 2 : ", list[2])

list[2] = 2001
print ("New value available at index 2 : ", list[2])

#delete list element
list = ['physics', 'chemistry', 1997, 2000]
print (list)

del list[2]
print ("After deleting value at index 2 : ", list)

list1 = ['physics', 'chemistry', 1997, 2000]


for i in list1: print(i)

list1, list2 = ['C++','Java', 'Python'], [456, 700, 200]
print ("Max value element : ", max(list1))
print ("Max value element : ", max(list2))

a = [i for i in range(5)]
print(a)

b = a.copy()
b[0] = 1
print(a)
print(b)

a = [i for i in range(5)]
b = tuple(a)
print(a)
