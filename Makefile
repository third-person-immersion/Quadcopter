all: main

main: main.o Object.o
	g++ main.o Object.o -o image-processing -I ./ -lopencv_highgui `pkg-config --cflags --libs opencv`

main.o: main.cpp
	g++ -c -std=c++11 -Wall main.cpp -I ./ -I ../cam-share

Object.o: Object.cpp
	g++ -c -std=c++11 -Wall Object.cpp

clean:
	rm -rf *o image-processing