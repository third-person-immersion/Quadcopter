#include <iostream>
#include <fstream>
#include <signal.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <boost/interprocess/shared_memory_object.hpp>

#include <Read.h>

using namespace std;
using namespace boost::interprocess;

/** Default constructor */
Read::Read(string logPath, string memoryName) {

    if(memoryName.empty())
    {
        memoryName = "cam-share-windows";
    }

    //Read resolution and memory size from file
    string line;
    ifstream infile;
    infile.open(logPath);
    if (infile.is_open()) {
        getline(infile, line);
        // width
        width=atoi(line.c_str());
        getline(infile, line);
        // height
        height=atoi(line.c_str());
        // memory size
        getline(infile, line);
        memorySize=atoi(line.c_str());
        infile.close();
    } else {
        cerr << "Unable to open file, using default resolution" << endl;
        height = 480, width = 640, memorySize = width * height * 4;
    }
        
    cout << "Starting Read 2" << endl;
    shared_memory_object shm (open_only, memoryName.c_str(), read_only);
    
    cout << "Starting Read 3" << endl;
    region = mapped_region(shm, read_only);
    
    cout << "Starting Read 4" << endl;
    frame = cv::Mat(height, width, CV_8UC3);
    
    cout << "Starting Read 5" << endl;
}

Read::~Read()
{
    // Free stuff?
}

cv::Mat Read::getFrame()
{
    frame.data = (uchar*)region.get_address();
    return frame;
}

int Read::getWidth()
{
    return width;
}

int Read::getHeight()
{
    return height;
}

int Read::getMemorysize()
{
    return memorySize;
}
