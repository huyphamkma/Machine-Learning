

data = "123"

#cach duyet 1
for i in data:
   print (i)

#cach duyet 2
for i in range(len(data)):
   print(data[i])


#
i = 0
while(i <= 5):
	print("count: ", i)
	i += 1
print("Good bye")

a = list(range(5))
print(a)


#
for i in a:
	print(i)


#####
for letter in 'Python':     # traversal of a string sequence
   print ('Current Letter :', letter)
print()


#####
fruits = ['banana', 'apple',  'mango']

for fruit in fruits:        # traversal of List sequence
   print ('Current fruit :', fruit)

print ("Good bye!")

#########
fruits = ['banana', 'apple',  'mango']
for index in range(len(fruits)):
   print ('Current fruit :', fruits[index])

print ("Good bye!")



##############
numbers = [11,33,55,39,55,75,37,21,23,41,13]

for num in numbers:
   if num%2 == 0:
      print ('the list contains an even number')
      break
else:
   print ('the list doesnot contain even number')

###############
import sys
for i in range(1,11):
   for j in range(1,11):
      k = i*j
      print (k, end = '\t')
   print()

 ############
for i in 'Python':    
   if i == 'h':
      break
   print ('Current Letter :', i)
  
var = 10                    # Second Example
while var > 0:              
   print ('Current variable value :', var)
   var = var -1
   if var == 5:
      break

print ("Good bye!")

####
num = int(input('Nhap 1 so bat ky: '))
number = [11,33,55,39,55,75,37,21,23,41,13]
for i in number:
	if(i == num):
		print('So da nhap co trong danh sach')
		break;
else:
	print('So da nhap khong co trong danh sach')

#####
for letter in 'Python':     # First Example
   if letter == 'h':
      continue
   print ('Current Letter :', letter)