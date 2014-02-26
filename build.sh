g++ -std=c++11 *.cpp -I ./ -lopencv_highgui `pkg-config --cflags --libs opencv`
./a.out
