
import pymysql
'''
db = pymysql.connect("localhost","root","","testdb")
cursor = db.cursor()
cursor.execute("SELECT VERSION()")
data = cursor.fetchone()
print ("Database version : %s " % data)

# disconnect from server
db.close()
'''


"""
#------CREATE TABLE-----------------

db = pymysql.connect("localhost","root","","testdb")
cursor = db.cursor();
cursor.execute("DROP TABLE IF EXISTS sinhvien1")

sql = '''CREATE TABLE sinhvien1 (first_name char(20) not null,
	last_name char(20),
	age int,
	sex char(1)
)'''
cursor.execute(sql)
db.close()
"""
"""
#----------------------INSERT DATABASE--------------------
db = pymysql.connect("localhost","root","","testdb")
cursor = db.cursor()

ten = input("Nhap ten: ")
ho = input("Nhap ho: ")
tuoi = int(input("Nhap tuoi: "))
gioitinh = input("Nhap gioitinh: ")

sql = '''INSERT INTO sinhvien1 (first_name, last_name, age, sex)
	values ('%s', '%s','%d', '%c')
''' %(ten, ho, tuoi, gioitinh)
try:
	cursor.execute(sql)
	db.commit()
except:
	db.rollback()

db.close()
"""
"""
#---------------------READ DATABASE-----------------------
db = pymysql.connect("localhost", "root", "", "testdb")
cursor = db.cursor()

sql = "SELECT * FROM sinhvien1"
try:
    cursor.execute(sql)
    results = cursor.fetchall()
    for row in results:
        ten = row[0]
        ho = row[1]
        tuoi = row[2]
        gioitinh = row[3]
        print('ten = %s\nho = %s\ntuoi = %d\ngioitinh = %c' %
              (ten, ho, tuoi, gioitinh))
except:
    print('ERROR: k doc dc du lieu')
db.close()
"""
"""
#------------------UPDATE DATABASE---------------------------
db = pymysql.connect("localhost","root","","testdb")
cursor = db.cursor()
tenmuonsua = input("Nhap ten muon sua: ")
tenmoi = input("Nhap ten moi")
sql = "UPDATE sinhvien1 SET first_name = '%s' WHERE first_name = '%s'" % (tenmoi, tenmuonsua)
try:
	cursor.execute(sql)
	db.commit()
except:
	db.rollback()
db.close()

"""
#----------------------DELETE---------------------------
db = pymysql.connect("localhost","root","","testdb")
cursor = db.cursor()
tenxoa = input("Nhap ten muon xoa: ")
sql = "DELETE FROM sinhvien1 WHERE first_name = '%s'" %(tenxoa)
try:
	cursor.execute(sql)
	db.commit()
except:
	db.rollback()
db.close()