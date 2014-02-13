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
int cP1 = 400; 
int cP2 = 30; //cerist papper
int maxHoughRadius = 80;

//Skillnad i höjd (y) mellan två punkter i centimeter (tre punkter i triangel används)
double twoPointsYDiff = 8.66;
double twoPointsXDiff = 10;
//pingisboll
//double ballRadius = 3.8;
//tennisboll
//double ballRadius = 6.7;
//Cerist papper stor
double ballRadius = 3.2;
//Cerist papper
//double ballRadius = 2.35;
//Ska vara två olika, men just nu testas med diagonalen, i grader
//inbyggd
//int FOV = 66;
//Microsoft webcam
double FOV = 73;

vector<Object> obs;
vector<Object> both;

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

	if(cP1 < 1)
	{
		cP1 = 1;
	}
	if(cP2 < 1)
	{
		cP2 = 1;
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
	sprintf_s(TrackbarName, "CIRCLE_RADIUS", maxHoughRadius);
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
	createTrackbar("CIRCLE_PARAM_1", trackbarWindowName, &cP1, 500, on_trackbar);
	createTrackbar("CIRCLE_PARAM_2", trackbarWindowName, &cP2, 100, on_trackbar);
	createTrackbar("CIRCLE_RADIUS", trackbarWindowName, &maxHoughRadius, 250, on_trackbar);
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

bool sorting(Object obj1, Object obj2)
{
	return obj1.getPrio() > obj2.getPrio();
}

void drawObject(vector<Object> objects, Mat &frame){
	int loops = 3;
	if(objects.size() < 3)
	{
		loops = objects.size();
	}
	for(int i = 0; i < loops; ++i)
	{
		int j = (i+1)%loops;
		cv::line(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), cv::Point(objects.at(j).getXPos(), objects.at(j).getYPos()),cv::Scalar(0, 0, 255));
	}
	for (int i = 0; i < objects.size(); i++){
		
		cv::circle(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), objects.at(i).getRadius(), cv::Scalar(0, 0, 255));
	}
}

double distanceFUCK(Object a, Object b){
	return sqrt((a.getXPos()-b.getXPos())*(a.getXPos()-b.getXPos()) + (a.getYPos()-b.getYPos())*(a.getYPos()-b.getYPos()));
}
double distanceFUCK(Object a, double X, double Y){
	return sqrt((a.getXPos()-X)*(a.getXPos()-X) + (a.getYPos()-Y)*(a.getYPos()-Y));
}
double radiusDiff(Object a, double radius){
	return abs(a.getRadius() - radius);
}

bool trackObjects(Mat &threshold, Mat &frame, Mat &gray) {
	double posX;
	double posY;
	int top = -1;
	int left = -1;
	int right = -1;
	bool found = false;
	Mat temp;
	threshold.copyTo(temp);
	Moments mom;
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Object> circles;
	vector<Vec3f> circlesTemp;



	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	//Check if any filtered objects was found
	if (hierarchy.size() > 0) {
		//If so, clear the object and "both" vector
		obs.clear(); // Clear the list of objects
		both.clear();
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
					//A true object is found
					found = true;
					//Add this object to the object vector
					Object o;
					o.setXPos(posX);
					o.setYPos(posY);
					o.setArea(area);
					o.setRadius(sqrt(area/PI));
					o.setPrio(0.0);
					o.setHasCircle(false);
					obs.push_back(o);
					//Draw the found object in dark red (this is not yet a object which should be tracked)
					Point center(cvRound(posX), cvRound(posY));
					int radius = cvRound(sqrt(area/PI));
					circle( frame, center, radius, Scalar(0,0,100));
				}
			}
		}
	}
	

	// Apply the Hough Transform to find the circles in the frame
	HoughCircles(gray, circlesTemp, CV_HOUGH_GRADIENT, 1, 80, cP1, cP2, 0, maxHoughRadius);

	cout << "Found circles: " << circlesTemp.size() << "\n";

	// Draw the detected circles
	for(int i = 0; i < circlesTemp.size(); i++ )
	{
		Point center(cvRound(circlesTemp[i][0]), cvRound(circlesTemp[i][1]));
		int radius = cvRound(circlesTemp[i][2]);
		// circle outline
		circle( frame, center, radius, Scalar(255,0,255), 3);
		//Convert the circles to objects and put them in the "both" vector
		Object o;
		o.setXPos(circlesTemp[i][0]);
		o.setYPos(circlesTemp[i][1]);
		o.setArea(circlesTemp[i][2]*circlesTemp[i][2]*PI);
		o.setRadius(circlesTemp[i][2]);
		o.setPrio(0.0);
		o.setHasCircle(false);
		circles.push_back(o);
	}
	
	//Match the objects with the circles and set the prio
	bool matchingCircles = false;
	int objectsWithCircle = 0;
	
	for(int i = 0; i < obs.size(); ++i) //Loop obs
	{
		for(int j = 0; j < circles.size(); ++j) //Loop circles
		{
			//Check if the radius and position of the circles and the filtered objects match
			if(obs.at(i).getXPos() - obs.at(i).getRadius()/2 <= circles.at(j).getXPos() && circles.at(j).getXPos() <= obs.at(i).getXPos() + obs.at(i).getRadius()/2 &&
				obs.at(i).getYPos() - obs.at(i).getRadius()/2 <= circles.at(j).getYPos() && circles.at(j).getYPos() <= obs.at(i).getYPos() + obs.at(i).getRadius()/2)
			{
				circles.at(j).setHasCircle(true); //temp
				//We have matching circles/filtered obejcts
				matchingCircles = true;
				//set the prio for the filtered objects
				obs.at(i).incPrio(abs(1/distanceFUCK(obs.at(i), circles.at(j).getXPos(), circles.at(j).getYPos())));
				obs.at(i).incPrio(abs(1/radiusDiff(obs.at(i), circles.at(j).getRadius())));
				obs.at(i).incPrio(2);
				//Say the filtered object has a corresponding circle
				obs.at(i).setHasCircle(true);
				objectsWithCircle++;
			}
		}
	}
	

	int tempAmount = 0;
	//Add the circles which does not belong to any filtered object into the "both" vector
	for (int i = 0; i < circles.size(); ++i)
	{
		if(!circles.at(i).getHasCircle())
		{
			tempAmount++;
			both.push_back(circles.at(i));
		}
		
	}
	cout << "Added circles: " << tempAmount << "\n";
	
	//Add all filtered objects to the "both" vector
	for (int i = 0; i < obs.size(); ++i)
	{
		both.push_back(obs.at(i));
	}

	//Calculate the avg radius of _all_ circles, as long as it has a corresponding circle
	double radiusAvg = 0;
	int matchedCircles = 0;

	for(int i = 0; i < both.size() && matchingCircles; ++i)
	{
		if(both.at(i).getHasCircle())
		{
			matchedCircles++;
			radiusAvg += both.at(i).getRadius();
		}
	}
	radiusAvg = radiusAvg/matchedCircles;

	cout << "radiusAvg: " << radiusAvg << "\n";

	int objectsWithNoPrio = 0;

	//Objects with prio != 0, check their radius and fix prio
	for(int i = 0; i < both.size() && matchingCircles; ++i)
	{
		if(both.at(i).getHasCircle())
		{
			objectsWithNoPrio++;
			both.at(i).incPrio(1/abs(both.at(i).getRadius() - radiusAvg));
		}
		stringstream ss;
		ss << both.at(i).getPrio();
		cv::putText(frame, ss.str(), cv::Point(both.at(i).getXPos(), both.at(i).getYPos()-40), 1, 1, cv::Scalar(0, 0, 170), 2);

	}
	
	
	cout << "objectsWithNoPrio: " << objectsWithNoPrio << "\n";
	
	
	//Sort the "both" vector with the highest prio first
	std::sort(both.begin(), both.end(), sorting);
	
	

	int dist = 0;
	if(objectsWithCircle >= 0 && both.size() >= 2)
	{

		dist = distanceFUCK(both.at(0), both.at(1));
		cout << "dist: " << dist << "\n";
		
		for(int i = 0; i < both.size(); ++i)
		{
			both.at(i).incPrio(1/(2*abs(dist - distanceFUCK(both.at(i), both.at(0)))));
			both.at(i).incPrio(1/(2*abs(dist - distanceFUCK(both.at(i), both.at(0)))));
		}
	}
	
	
	//Sort the "both" vector with the highest prio first
	// must be done again since the loop above can fuck it all up
	std::sort(both.begin(), both.end(), sorting);
	

	//If "both" vector is of size >= 3, then only loop 3 times (showing only the 3 circles with highest prio)
	int loops = 3;
	if(both.size() < 3)
	{
		loops = both.size();
	}

	cout << "\n\nNumber of hits obs:  " << obs.size() << "\n";
	cout << "\n\nNumber of hits both: " << both.size() << "\n\n";
	//Calculate the 3D position of the 3 circles with highest prio.
	for(int i=0; i<loops; i++){
		double zDistance = ballRadius * frame.cols / (2*both.at(i).getRadius()*tan(FOV * PI/360));
		stringstream ss;
		ss << zDistance;
		cv::putText(frame, ss.str(), cv::Point(both.at(i).getXPos(), both.at(i).getYPos()), 2, 0.5, cv::Scalar(0,100,0));
		stringstream ss2;
		ss2 << both.at(i).getRadius();
		cv::putText(frame, ss2.str(), cv::Point(both.at(i).getXPos(), both.at(i).getYPos()+40), 1, 1, cv::Scalar(0,255,0));
	}



	//Allt detta är för koordinatsystemet. Ja.
	if(both.size()>=3){
		double position[3][3];
		for(int i=0; i<3; i++){
			position[i][0] = (frame.cols - both.at(i).getXPos()) * ballRadius/both.at(i).getRadius();
			position[i][1] = (frame.rows - both.at(i).getYPos()) * ballRadius/both.at(i).getRadius();
			position[i][2] = ballRadius * frame.rows / (2*both.at(i).getRadius()*tan(FOV * PI/360));
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
		double w = both.at(0).getXPos();
		double h = both.at(0).getYPos();

		//Koordinatsystemsgrejen (suger)
		cv::line(frame, cv::Point(w, h),cv::Point(w + vect1[0], h + vect1[1]),cv::Scalar(255,0,0), 3);
		cv::line(frame, cv::Point(w, h),cv::Point(w + vect2[0], h + vect2[1]),cv::Scalar(0,255,0), 3);
		cv::line(frame, cv::Point(w, h),cv::Point(w + vect3[0], h + vect3[1]),cv::Scalar(0,0,255), 3);
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
	//Sleep(1000);

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

	/** Microsoft webcam
	    Blått papper
		YCrCp färger
		*/
	H_MIN = 74;
	H_MAX = 163;
	S_MIN = 133;
	S_MAX = 184;
	V_MIN = 0;
	V_MAX = 256;
	MINAREA = 750;
	ERODE = 13;
	DILATE = 13;

	/* Microsoft webcam */
	/*CERIST PAPPER
	H_MIN = 112;
	H_MAX = 195;
	S_MIN = 128;
	S_MAX = 196;
	V_MIN = 154;
	V_MAX = 254;
	MINAREA = 370;
	cP1 = 300;
	ERODE = 3;
	DILATE = 4; //7*/
	
	
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
	/* inbyggd kamera tennisboll
	
	H_MIN = 68;
	H_MAX = 98;
	S_MIN = 46;
	S_MAX = 158;
	V_MIN = 82;
	V_MAX = 256;
	MINAREA = 1500;
	ERODE = 4;
	DILATE = 5;*/
	
	
	createTrackbars();

	//Sleep(1000);

	CvCapture* capture = cvCreateCameraCapture(0);


	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 720);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1280);

	while (1) {
		cam.read(frame);

		// Blur the image a bit
		GaussianBlur(frame, frame, Size(3, 3), 0, 0);

		// Convert from RGB to HSV
		cvtColor(frame, hsvFrame, COLOR_RGB2YCrCb);

		// Convert to binary B&W
		inRange(hsvFrame, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);

		//Gaussian the black/white image
		GaussianBlur(threshold, threshold, Size(3, 3), 0, 0);
		//Convert to grayscale
		cvtColor(frame, gray, CV_BGR2GRAY);
		//Gaussian the grey image
		GaussianBlur(gray, gray, Size(3, 3), 0, 0);


		//morphops the binary image
		morphOps(threshold);

		if (!frame.empty()){
			try {

				// Tracking	
				bool found = trackObjects(threshold, frame, gray);
				if (found) {
					drawObject(both, frame);
				}

				//Canny(gray, gray, cP1/3, cP1);
				

				// Display image
				imshow("Image", frame);
				//imshow("YCrCb image", hsvFrame);
				imshow("Binary image", threshold);
				//imshow("Gray image", gray);
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



