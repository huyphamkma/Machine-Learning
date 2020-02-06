#!C:\Users\huyla\AppData\Local\Programs\Python\Python36-32/python.exe
'''
print ("Content-type:text/html\r\n\r\n")
print ('<html>')
print ('<head>')
print ('<title>Hello Word - First CGI Program</title>')
print ('</head>')
print ('<body>')
print ('<h2>Hello Word! This is my first CGI program</h2>')
print ('</body>')
print ('</html>')

'''
'''
# Import cac module de xu ly CGI 
import cgi, cgitb 

# Tao instance cua FieldStorage 
form = cgi.FieldStorage() 

# Lay du lieu tu cac truong
first_name = form.getvalue('first_name')
last_name  = form.getvalue('last_name')

print ("Content-type:text/html\r\n\r\n")
print ("<html>")
print ("<head>")
print ("<title>Hello - Second CGI Program</title>")
print ("</head>")
print ("<body>")
print ("<h2>Hello %s %s</h2>" % (first_name, last_name))
print ("</body>")
print ("</html>")

'''

import cgi, cgitb 
# Create instance of FieldStorage 

form = cgi.FieldStorage() 

# Get data from fields
first_name = form.getvalue('first_name')
last_name  = form.getvalue('last_name')

print ("Content-type:text/html\r\n\r\n")
print ("<html>")
print ("<head>")
print ("<title>Hello - Second CGI Program</title>")



print ("</head>")
print ("<body>")
print ("<h2>Hello %s %s</h2>" % (first_name, last_name))
print ("</body>")
print ("</html>")

print ("""<form action = "/cgi-bin/CGIProgramming.py" method = "post">
First Name: <input type = "text" name = "first_name"><br />
Last Name: <input type = "text" name = "last_name" />

<input type = "submit" value = "Submit" />
</form>
""")

#--------------Checkbox--------------------
print("""<form action = "/cgi-bin/CGIProgramming.py" method = "POST" target = "_blank">
   <input type = "checkbox" name = "maths" value = "on" /> Maths
   <input type = "checkbox" name = "physics" value = "on" /> Physics
   <input type = "submit" value = "Select Subject" />
</form>
""")


if form.getvalue('maths'):
   math_flag = "ON"
else:
   math_flag = "OFF"

if form.getvalue('physics'):
   physics_flag = "ON"
else:
   physics_flag = "OFF"


print ("<html>")
print ("<head>")
print ("<title>Vi du Radio button</title>")
print ("</head>")
print ("<body>")
print ("CheckBox Maths is : %s</h2>" % math_flag)
print ("CheckBox Physics is : %s</h2>" % physics_flag)
print ("</body>")
print ("</html>")

#----------------Radio box---------------------
print("""<form action = "/cgi-bin/CGIProgramming.py" method = "post" target = "_blank">
   <input type = "radio" name = "subject" value = "maths" /> Maths
   <input type = "radio" name = "subject" value = "physics" /> Physics
   <input type = "submit" value = "Select Subject" />
</form>
""")


if form.getvalue('subject'):
   subject = form.getvalue('subject')
else:
   subject = "Not set"


print("<html>")
print("<head>")
print("<title>Radio - Fourth CGI Program</title>")
print("</head>")
print("<body>")
print("<h2> Selected Subject is %s</h2>" % subject)
print("</body>")
print("</html>")


#-------------text area---------------------
print("""<form action = "/cgi-bin/CGIProgramming.py" method = "post" target = "_blank">
   <textarea name = "textcontent" cols = "40" rows = "4">
      Type your text here...
   </textarea>
   <input type = "submit" value = "Submit" />
</form>
""")


if form.getvalue('textcontent'):
   text_content = form.getvalue('textcontent')
else:
   text_content = "Not entered"


print ("<html>")
print ("<head>")
print ("<title>Text Area - Fifth CGI Program</title>")
print ("</head>")
print ("<body>")
print ("<h2> Entered Text Content is %s</h2>" % text_content)
print ("</body>")

#--------------------drop down box--------------------
print("""<form action = "/cgi-bin/CGIProgramming.py" method = "post" target = "_blank">
   <select name = "dropdown">
      <option value = "Maths" selected>Maths</option>
      <option value = "Physics">Physics</option>
   </select>
   <input type = "submit" value = "Submit"/>
</form>
""")

#form = cgi.FieldStorage() 

# Get data from fields
if form.getvalue('dropdown'):
   subject = form.getvalue('dropdown')
else:
   subject = "Not entered"


print ("<html>")
print ("<head>")
print ("<title>Text Area - Fifth CGI Program</title>")
print ("</head>")
print ("<body>")
print ("<h2> Selected Subject is %s</h2>" % subject)
print ("</body>")

#-------------file upload------------------
print("""<html>
   <body>
      <form enctype = "multipart/form-data" action = "CGIProgramming.py.py" method = "post">
      <p>File: <input type = "file" name = "filename" /></p>
      <p><input type = "submit" value = "Upload" /></p>
      </form>
   </body>
</html>
""")

fileitem = form['filename']

# Test if the file was uploaded
if fileitem.filename:
   # strip leading path from file name to avoid 
   # directory traversal attacks
   fn = os.path.basename(fileitem.filename)
   open('/tmp/' + fn, 'wb').write(fileitem.file.read())

   message = 'The file "' + fn + '" was uploaded successfully'
   
else:
   message = 'No file was uploaded'
   
print ("""\
Content-Type: text/html\n
<html>
   <body>
      <p>%s</p>
   </body>
</html>
""" % (message,))