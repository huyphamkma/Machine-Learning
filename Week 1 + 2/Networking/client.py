import socket

#tao 1 doi tuong socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
port = 1234
s.connect((host, port))
msg = s.recv(1024)
s.close()
print(msg.decode('ascii'))