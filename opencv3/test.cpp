// OpenCV webcam test.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <windows.h>
#include "Object.h"
#define PI 3.14159265

using namespace cv;
using namespace std;

int MINAREA = 2000;
int ERODE = 30;
int DILATE = 60;
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
int cP1 = 100;
int cP2 = 25;

//Skillnad i höjd (y) mellan två punkter i centimeter (tre punkter i triangel används)
double twoPointsYDiff = 8.66;
double twoPointsXDiff = 10;
//pingisboll
//double ballRadius = 3.8;
//tennisboll
double ballRadius = 6.7;
//Ska vara två olika, men just nu testas med diagonalen, i grader
//inbyggd
//int FOV = 66;
//Microsoft webcam
double FOV = 70;

vector<Object> obs;

void on_trackbar(int, void*)
{//This function gets called whenever a
	// trackbar position is changed
	if(ERODE < 1)
	{
		ERODE = 1;
	}

	if(DILATE < 1)
	{
		DILATE = 1;
	}
}

void createTrackbars(){
	string trackbarWindowName = "trackbars";
	//create window for trackbars
	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf_s(TrackbarName, "H_MIN", H_MIN);
	sprintf_s(TrackbarName, "H_MAX", H_MAX);
	sprintf_s(TrackbarName, "S_MIN", S_MIN);
	sprintf_s(TrackbarName, "S_MAX", S_MAX);
	sprintf_s(TrackbarName, "V_MIN", V_MIN);
	sprintf_s(TrackbarName, "V_MAX", V_MAX);
	sprintf_s(TrackbarName, "MINAREA", MINAREA);
	sprintf_s(TrackbarName, "ERODE", ERODE);
	sprintf_s(TrackbarName, "DILATE", DILATE);
	sprintf_s(TrackbarName, "CIRCLE_PARAM_1", cP1);
	sprintf_s(TrackbarName, "CIRCLE_PARAM_2", cP2);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, 256, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, 256, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, 256, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, 256, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, 256, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, 256, on_trackbar);
	createTrackbar("MINAREA", trackbarWindowName, &MINAREA, 3000, on_trackbar );
	createTrackbar("ERODE", trackbarWindowName, &ERODE, 20, on_trackbar);
	createTrackbar("DILATE", trackbarWindowName, &DILATE, 20, on_trackbar);
	createTrackbar("CIRCLE_PARAM_1", trackbarWindowName, &cP1, 400, on_trackbar);
	createTrackbar("CIRCLE_PARAM_2", trackbarWindowName, &cP2, 400, on_trackbar);
}

void morphOps(Mat &thresh){

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(ERODE, ERODE));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(DILATE, DILATE));


	// Removes small white areas, like noice etc
	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);

	// Makes the remaining whitespace larger
	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}

void drawObject(vector<Object> objects, Mat &frame){
	for (int i = 0; i < objects.size(); i++){
		int j = (i+1)%objects.size();
		cv::circle(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), objects.at(i).getRadius(), cv::Scalar(0, 0, 255));
		cv::line(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), cv::Point(objects.at(j).getXPos(), objects.at(j).getYPos()),cv::Scalar(0, 0, 255));
	}
}

double Distance(Object a, Object b){
	return sqrt((a.getXPos()-b.getXPos())*(a.getXPos()-b.getXPos()) + (a.getYPos()-b.getYPos())*(a.getYPos()-b.getYPos()));
}

bool trackObjects(Mat &threshold, Mat &frame) {
	double posX;
	double posY;
	int top = -1;
	int left = -1;
	int right = -1;
	bool found = false;
	//Temp av threshold (den svarta bilden)
	Mat temp;
	Moments mom;
	threshold.copyTo(temp);
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	if (hierarchy.size() > 0) {
		obs.clear(); // Clear the list of objects
		int numObjects = hierarchy.size();
		for (int index = 0; index >= 0; index = hierarchy[index][0]) {
			mom = moments((Mat)contours[index], 1);
			double moment10 = mom.m10;
			double moment01 = mom.m01;
			double area = mom.m00;

			if (area > MINAREA) {
				posX = moment10 / area;
				posY = moment01 / area;

				

				if (posX >= 0 && posY >= 0) {
					found = true;
					Object o;
					o.setXPos(posX);
					o.setYPos(posY);
					o.setArea(area);
					o.setRadius(sqrt(area/PI));
					obs.push_back(o);
				}
			}
		}
	}

	//cout << "framerows: " << frame.rows << "\n";
	//cout << "framecolumns: " << frame.cols << "\n";
	cout << "\n\nNumber of hits: " << obs.size() << "\n\n";
	for(int i=0; i<obs.size(); i++){
		double zDistance = ballRadius * frame.rows / (2*obs.at(i).getRadius()*tan(FOV * PI/360));
		stringstream ss;
		ss << zDistance;
		cv::putText(frame, ss.str(), cv::Point(obs.at(i).getXPos(), obs.at(i).getYPos()), 2, 2, cv::Scalar(0,255,0));
		stringstream ss2;
		ss2 << obs.at(i).getRadius();
		cv::putText(frame, ss2.str(), cv::Point(obs.at(i).getXPos(), obs.at(i).getYPos()+40), 1, 1, cv::Scalar(0,255,0));
	}

	if(obs.size()==3){
		double position[3][3];
		for(int i=0; i<3; i++){
			position[i][0] = (frame.cols - obs.at(i).getXPos()) * ballRadius/obs.at(i).getRadius();
			position[i][1] = (frame.rows - obs.at(i).getYPos()) * ballRadius/obs.at(i).getRadius();
			position[i][2] = ballRadius * frame.rows / (2*obs.at(i).getRadius()*tan(FOV * PI/360));
		}
		
		double vect1[3];
		double vect2[3];
		double vect3[3];

		double lineLength = 40;
		
		for(int i=0; i<3; i++){
			vect1[i] = (position[0][i] - position[1][i]);
		}
		double length = sqrt(vect1[0]*vect1[0] + vect1[1]*vect1[1] + vect1[2]*vect1[2]);
		for(int i=0; i<3; i++){
			vect1[i] = lineLength * vect1[i]/length;
		}
		for(int i=0; i<3; i++){
			vect2[i] = (position[0][i] - position[2][i]);
		}
		length = sqrt(vect2[0]*vect2[0] + vect2[1]*vect2[1] + vect2[2]*vect2[2]);
		for(int i=0; i<3; i++){
			vect2[i] = lineLength * vect2[i]/length;
		}
		vect3[0] = (vect1[1]*vect2[2] - vect1[2]*vect2[1]);
		vect3[1] = (vect1[2]*vect2[0] - vect1[0]*vect2[2]);
		vect3[2] = (vect1[0]*vect2[1] - vect1[1]*vect2[0]);
		length = sqrt(vect3[0]*vect3[0] + vect3[1]*vect3[1] + vect3[2]*vect3[2]);
		for(int i=0; i<3; i++){
			vect3[i] = lineLength * vect3[i]/length;
		}
		double w = frame.cols/2;
		double h = frame.rows/2;
		cv::line(frame, cv::Point(w, h),cv::Point(w + vect1[0], h + vect1[1]),cv::Scalar(255,0,0));
		cv::line(frame, cv::Point(w, h),cv::Point(w + vect2[0], h + vect2[1]),cv::Scalar(0,255,0));
		cv::line(frame, cv::Point(w, h),cv::Point(w + vect3[0], h + vect3[1]),cv::Scalar(0,0,255));
	}

	/*///////////////////////////////////////////////////////////
	if (obs.size() == 3) {
		for (int i = 0; i < obs.size(); i++){
			if (obs.at(i).getYPos() <= obs.at(0).getYPos()
				&& obs.at(i).getYPos() <= obs.at(1).getYPos()
				&& obs.at(i).getYPos() <= obs.at(2).getYPos()) {
				obs.at(i).setType("Top");
				top = i;
				//cout << "Position of top object: X: " << obs.at(i).getXPos() << ", Y: " << obs.at(i).getYPos() << "\n";
			}
			else if (obs.at(i).getXPos() <= obs.at(0).getXPos()
				&& obs.at(i).getXPos() <= obs.at(1).getXPos()
				&& obs.at(i).getXPos() <= obs.at(2).getXPos())  {
				obs.at(i).setType("Left");
				left = i;
				//cout << "Position of left object: X: " << obs.at(i).getXPos() << ", Y: " << obs.at(i).getYPos() << "\n";
			}
			else {
				obs.at(i).setType("Right");
				right = i;
				//cout << "Position of right object: X: " << obs.at(i).getXPos() << ", Y: " << obs.at(i).getYPos() << "\n";
			}
		}

		

		// Calculate if copter should move
		if (top != -1 && left != -1 && right != -1) {
			double yDiff = obs.at(left).getYPos() - obs.at(top).getYPos();
			double xDiff = obs.at(right).getXPos() - obs.at(left).getXPos();

			//Beräkning av avstånd
			double xDistance = frame.cols/2 - obs.at(left).getXPos() - xDiff/2;
			double yDistance = frame.rows/2 - obs.at(left).getYPos() + yDiff/2;
			yDistance *= twoPointsYDiff/yDiff;
			xDistance *= twoPointsYDiff/yDiff;
			double zDistance = twoPointsYDiff * frame.rows / (yDiff*tan(FOV * PI/360));

			cout << "Distance X: " << xDistance << "cm\n";
			cout << "Distance Y: " << yDistance << "cm\n";
			cout << "Distance Z: " << zDistance << "cm\n";
			double triangleMidX = obs.at(left).getXPos() + xDiff/2;
			double triangleMidY = obs.at(top).getYPos() + yDiff/2;
			cv::line(frame, cv::Point(triangleMidX, triangleMidY), cv::Point(frame.cols/2, frame.rows/2), cv::Scalar(0, 0, 0));

			// rotation kring Y givet ingen annan rotation
			double yRot = asin(xDiff/(yDiff*twoPointsXDiff/twoPointsYDiff));
			if(obs.at(left).getXPos()-obs.at(top).getXPos() < obs.at(top).getXPos()-obs.at(right).getXPos())
			{
				yRot = 90 - yRot;
			}
			else
			{
				yRot -= 90;
			}
			// rotation kring Z givet ingen annan rotation
			double zRot = -atan((obs.at(left).getYPos()-obs.at(right).getYPos())/xDiff);

			cout << "Rotation Y: " << yRot * 180/PI << "degrees\n";
			cout << "Rotation Z: " << zRot * 180/PI << "degrees\n";
			double axleLength = 40;
			//X axle
			double xAxleLength = axleLength * sin(PI/2-yRot);
			double endPointX = triangleMidX + xAxleLength * cos(zRot);
			double endPointY = triangleMidY + xAxleLength * sin(zRot);
			cv::line(frame, cv::Point(triangleMidX, triangleMidY), cv::Point(endPointX, endPointY), cv::Scalar(255, 0, 0));
			//Y axle
			endPointX = triangleMidX - axleLength * sin(zRot);
			endPointY = triangleMidY + axleLength * cos(zRot);
			cv::line(frame, cv::Point(triangleMidX, triangleMidY), cv::Point(endPointX, endPointY), cv::Scalar(0, 255, 0));
			//Z axle
			endPointX = triangleMidX + axleLength * cos(yRot);
			//cv::line(frame, cv::Point(frame.cols/2, frame.rows/2), cv::Point(endPointX, frame.rows/2), cv::Scalar(0, 0, 255));

			// mer rotations beräkningar
			double lTD = Distance(obs.at(left),obs.at(top));
			double tRD = Distance(obs.at(top),obs.at(right));
			double rLD = Distance(obs.at(right),obs.at(left));
			cout <<"\n";
			cout << "lTD: " << lTD << "px\n";
			cout << "tRD: " << tRD << "px\n";
			cout << "rLD: " << rLD << "px\n";
		}
	}
	*/////////////////////////////////////////////////////////////////////////////////////////
	return found;
}

int main(int argc, char** argv)
{
	VideoCapture cam(0);
	Sleep(1000);

	if (!cam.isOpened()) {
		cout << "Error loading camera";
	}
	else {
		cout << "Camera loaded OK\n\n";
	}

	//Själva "bilden" (den med färg o shiet)
	Mat frame;
	Mat hsvFrame;
	//svartvita bilden (den filtrerade)
	Mat threshold;
	//grayscale image
	Mat gray;


	/* Microsoft webcam 
	H_MIN = 112;
	H_MAX = 195;
	S_MIN = 137;
	S_MAX = 196;
	V_MIN = 154;
	V_MAX = 254;
	MINAREA = 370;
	ERODE = 3;
	DILATE = 7;
	*/
	/* inbyggd kamera
	H_MIN = 35;
	H_MAX = 68;
	S_MIN = 98;
	S_MAX = 170;
	V_MIN = 63;
	V_MAX = 256;
	MINAREA = 300;
	ERODE = 1;
	DILATE = 5;
	*/
	/* inbyggd kamera pingisboll
	H_MIN = 32;
	H_MAX = 89;
	S_MIN = 70;
	S_MAX = 189;
	V_MIN = 86;
	V_MAX = 256;
	MINAREA = 3000;
	ERODE = 4;
	DILATE = 5;
	*/
	/* inbyggd kamera tennisboll*/
	H_MIN = 68;
	H_MAX = 98;
	S_MIN = 46;
	S_MAX = 158;
	V_MIN = 82;
	V_MAX = 256;
	MINAREA = 1500;
	ERODE = 4;
	DILATE = 5;
	
	
	createTrackbars();

	Sleep(1000);

	CvCapture* capture = cvCreateCameraCapture(0);


	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 720);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1280);

	while (1) {
		cam.read(frame);

		// Blur the image a bit
		GaussianBlur(frame, frame, Size(3, 3), 0, 0);

		// Convert from RGB to HSV
		cvtColor(frame, hsvFrame, COLOR_RGB2HSV);

		// Convert to binary B&W
		inRange(hsvFrame, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);

		GaussianBlur(threshold, threshold, Size(3, 3), 0, 0);


		//morphops the binary image
		morphOps(threshold);

		if (!frame.empty()){
			try {
				// Tracking	
				bool found = trackObjects(threshold, frame);
				if (found) {
					drawObject(obs, frame);
				}

				////////////////Circle detection////////////////////////////

				//Convert to grayscale
				//cvtColor(frame, gray, CV_BGR2GRAY);
				gray = threshold;
				
				GaussianBlur(gray, gray, Size(3, 3), 0, 0);

				vector<Vec3f> circles;

				/// Apply the Hough Transform to find the circles
				HoughCircles( gray, circles, CV_HOUGH_GRADIENT, 1, 40, cP1, cP2, 30, 100);

				/// Draw the circles detected
				 for( size_t i = 0; i < circles.size(); i++ )
				{
					  Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
					  int radius = cvRound(circles[i][2]);
					 // circle outline
					 circle( frame, center, radius, Scalar(255,0,0));
				 }

				// Display image
				imshow("Image", frame);
				imshow("HSV image", hsvFrame);
				imshow("Binary image", threshold);
			}
			catch (cv::Exception & e)
			{
				cout << e.what() << endl;
			}
		}
		if (waitKey(30) >= 0) break;
	}

	return 0;
}



