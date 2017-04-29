from time import time

class Timer(object):
    def __init__(self):
	self.start = time()

    def interval(self, msg):
	current = time()
	elapsed = int(current - self.start)
	self.start = current
	minutes = elapsed / 60
	seconds = elapsed % 60
	print "{} - Minutes: {}, Seconds: {}".format(msg, minutes, seconds)
	

