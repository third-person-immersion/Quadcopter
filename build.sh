g++ -std=c++11 *.cpp -I./ -I../cam-share -lopencv_highgui `pkg-config --cflags --libs opencv` -o quadcopter
