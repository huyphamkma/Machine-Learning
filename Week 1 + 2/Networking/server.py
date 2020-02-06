import socket

#tao 1 doi tuong socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#lay ten thiet bi local
host = socket.gethostname()
#dat cong
port = 1234
#ket noi toi port
server_socket.bind((host, port))
#tao 5 request
server_socket.listen(5)
while True:
	#thiet lap 1 ket noi
	client_socket, addr = server_socket.accept()
	print("Co 1 ket noi tu %s" %(str(addr)))
	msg = "Thank you for connecting \n"
	client_socket.send(msg.encode('ascii'))
	client_socket.close()