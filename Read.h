#ifndef Read_h
#define Read_h


#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <string>

using namespace std;
using namespace boost::interprocess;

class Read
{
public:
    cv::Mat frame;
    mapped_region region;
    Read(string logPath, string memoryName);
    ~Read();
    cv::Mat Read::getFrame();
    int getWidth();
    int getHeight();
    int getMemorysize();
private:
    int width, height, memorySize;
};

#endif