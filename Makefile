all: main

main: main.o Object.o Read_unix.o
	g++ main.o Object.o Read_unix.o -o image-processing -I ./ -lopencv_highgui `pkg-config --cflags --libs opencv`

Read_unix.o: ../cam-share/Read_unix.cpp
	g++ -c -std=c++11 -Wall ../cam-share/Read_unix.cpp -I../cam-share

main.o: main.cpp
	g++ -c -std=c++11 -Wall main.cpp -I ./ -I ../cam-share

Object.o: Object.cpp
	g++ -c -std=c++11 -Wall Object.cpp

clean:
	rm -rf *o image-processing