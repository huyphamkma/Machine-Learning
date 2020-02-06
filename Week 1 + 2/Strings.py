var1 = 'Hello World!'
var2 = "Python Programming"

print("var1[0]: ", var1[0])
print("var2[1:5]: ", var2[1:5])

#update strings
var1 = 'Hello World!'
print ("Updated String :- ", var1[:6] + 'Python')

para_str = """this is a long string that is made up of
several lines and non-printable characters such as
TAB ( \t ) and they will show up that way when displayed.
NEWLINEs within the string, whether explicitly given like
this within the brackets [ \n ], or just a NEWLINE within
the variable assignment will also show up.
"""
print (para_str)

print("My name is %s and weight is %d kg" % ('Zara', 21))
print("My name is {} and weight is {} kg" .format('Zara', 21))

print ('C:\\nowhere')
print (r'C:\\nowhere')

# Thêm các ký tự a vào đầu và cuối chuỗi để chuỗi có độ dài là 40
str = "this is string example....wow!!!"
print ("str.center(40, 'a') : ", str.center(40, 'a'))

print(f"String: {str}")

#cat chuoi thanh list
a = 'ha he hi ho'
b = a.split(' ')
print(b)

#cat chuoi thanh tuple
b = a.partition(' ')
print(b)


