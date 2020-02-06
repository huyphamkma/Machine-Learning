import _thread
import time

'''
def print_time(theardName, delay):
	count = 0
	while count < 5:
		time.sleep(delay)
		count += 1
		print("%s: %s" % (theardName, time.ctime(time.time())))

try:
	_thread.start_new_thread(print_time, ("Thread-1",2,))
	_thread.start_new_thread(print_time, ("Thread-2",4, ))
except:
	print ("Error")
'''

import threading

class myThread (threading.Thread):
	def __init__(self, name, counter, delay):
		threading.Thread.__init__(self)
		self.name = name
		self.counter = counter
		self.delay = delay

	def run(self):
		print("Starting ", self.name)
		threadLock.acquire()
		while self.counter > 0:
			time.sleep(self.delay)
			print("%s : %s" % (self.name, time.ctime(time.time())))
			self.counter -= 1
		print("Exiting ", self.name)
		threadLock.release()
		

threadLock = threading.Lock()
thread = []
thread1 = myThread("Thread 1", 5, 2)
thread2 = myThread("Thread 2", 5, 3)


thread1.start()
thread2.start()
thread.append(thread1)
thread.append(thread2)
for i in thread:
	i.join()
print("Exiting main thread")