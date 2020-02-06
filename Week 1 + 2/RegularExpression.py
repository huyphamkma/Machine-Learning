import re
'''
pattern = '^a...s$'
str = input('Nhap 1 chuoi: ')
result = re.match(pattern, str)
if(result):
	print("Hop le")
else:
	print('Khong hop le')
'''

print('---------------re.findall(pattern, string)-------------')
str = 'Hello 12 hi 89. Howdy 34'
pattern = '\d+'
print(re.findall(pattern, str))

print('---------------re.split(pattern, string, maxsplit)-------------')
str = 'The rain in Vietnam'
pattern = '\s'
print(re.split(pattern, str))
print(re.split(pattern, str, 1))


print('---------------re.sub(pattern, replace, string, count)-------------')
str = 'abc 12\
de 23 \n f456'
pattern = '\s+'
replace = ''
print(re.sub(pattern, replace, str))
print(re.sub(pattern, replace, str, 2))

print('---------------re.subn(pattern, replace, string)-------------')
str = 'abc 123 456 def'
pattern = '\s'
replace = ''
print(re.subn(pattern, replace, str))

print('---------------re.search(pattern, string)-------------')
str = 'abc def ghi jkl'
pattern = '\Aabc'
print(re.search(pattern, str))
if(re.search(pattern, str)):
	print('abc nam o dau cau')
else:
	print('abc k nam o dau cau')


print('---------------match.group()-------------')
str = '39801 355, 2102 123'
pattern = '(\d{3}) (\d{2})'
match = re.search(pattern, str)
print('match.group(): ',match.group())
print('match.group(1): ',match.group(1))
print('match.group(2): ',match.group(2))
print('match.start(): ',match.start())
print('match. end(): ',match.end())
print('match.span(): ',match.span())
print('match.re: ', match.re)
print('match.string: ', match.string)

print('---------------r-------------')
str = '\naaa and \r are escape sequences.'
#pattern = '[\n\r]'
print(re.findall(r'[\n\r]', str))