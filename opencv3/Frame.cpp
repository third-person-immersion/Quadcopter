#include "Frame.h"

Frame::Frame(void)
{
}


Frame::~Frame(void)
{
}


Mat Frame::getFrame(){

	return Frame::frame;

}

void Frame::setFrame(Mat frame){

	Frame::frame = frame;

}


double Frame::getPrio(){
	return Frame::prio;
}

void Frame::incPrio(double prio){
	Frame::prio+=prio;
}

void Frame::decPrio(double prio){
	Frame::prio-=prio;
}

void Frame::setPrio(double prio){
	Frame::prio = prio;
}
