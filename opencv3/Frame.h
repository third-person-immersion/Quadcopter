#pragma once
#include <string>
#include <opencv2/core/core.hpp>
using namespace cv;

class Frame
{
public:
	Frame(void);
	~Frame(void);

	Mat getFrame();
	void setFrame(Mat frame);

	double getPrio();
	void incPrio(double prio);
	void decPrio(double prio);
	void setPrio(double prio);

private:

	double prio;
	Mat frame;

};