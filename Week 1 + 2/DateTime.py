import time;

ticks = time.time()
print ("Number of ticks since 12:00am, January 1, 1970:", ticks)

print(time.localtime())

localtime = time.asctime(time.localtime(time.time()))
print("thoi gian hien tai: ", localtime)


import calendar
cal = calendar.month(2020, 2)
print("Lich thang 2 nam 2020:")
print(cal)


cal = calendar.monthcalendar(2020, 2)
print (cal)

for thang in range(1, 13):
	count = 0
	for tuan in calendar.monthcalendar(2020, thang):
		for ngay in tuan:
			if(ngay != 0):
				count += 1;
	print ('Thang ', thang, ' co ', count, ' ngay')



print ("Start : %s" % time.ctime())
time.sleep( 5 )
print ("End : %s" % time.ctime())