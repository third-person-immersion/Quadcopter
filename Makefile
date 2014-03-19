all: main

main: test.o Object.o
	g++ test.o Object.o -o image-processing -I ./ -lopencv_highgui `pkg-config --cflags --libs opencv`

test.o: test.cpp
	g++ -c -std=c++11 -Wall test.cpp -I ./ -I../cam-share

Object.o: Object.cpp
	g++ -c -std=c++11 -Wall Object.cpp

clean:
	rm -rf *o image-processing