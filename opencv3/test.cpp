// OpenCV webcam test.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <windows.h>
#include "Frame.h"
//#include <Getopt.h>
#include <stdio.h>
//#include <unistd.h>
#include "Object.h"
#define PI 3.14159265

using namespace cv;
using namespace std;

int MAX_DISTANCE_BETWEEN_CIRCLES = 30;
int MIN_DISTANCE_BETWEEN_CIRCLES = 10;
int MINAREA = 2000;
int MAXAREA = 15000;
int ERODE = 30;
int DILATE = 60;
int Y_MIN = 0;
int Y_MAX = 256;
int Cb_MIN = 133;
int Cb_MAX = 184;
int Cr_MIN = 0;
int Cr_MAX = 256;
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
int cP1 = 400; 
int cP2 = 30; //cerist papper
int maxHoughRadius = 80;

//The radius of the current ball used
double ballRadius = 7.5/2;
//The Field Of View of the camera used at the moment
//Microsoft webcam
int FOV = 48;//66;



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


void createTrackbars(int number, char* min1, char* max1, char* min2, char* max2, char* min3, char* max3, int *minInt1, int *maxInt1, int *minInt2, int *maxInt2, int *minInt3, int *maxInt3){
    std::stringstream sstm;
    sstm << "Trackbars" << number;
    string trackbarWindowName = sstm.str();
    //create window for trackbars
    namedWindow(trackbarWindowName, 0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf_s(TrackbarName, min1, minInt1);
    sprintf_s(TrackbarName, max1, maxInt1);
    sprintf_s(TrackbarName, min2, minInt2);
    sprintf_s(TrackbarName, max2, maxInt2);
    sprintf_s(TrackbarName, min3, minInt3);
    sprintf_s(TrackbarName, max3, maxInt3);
    sprintf_s(TrackbarName, "MINAREA", MINAREA);
    sprintf_s(TrackbarName, "MAXAREA", MAXAREA);
    sprintf_s(TrackbarName, "ERODE", ERODE);
    sprintf_s(TrackbarName, "DILATE", DILATE);
    sprintf_s(TrackbarName, "CIRCLE_PARAM_1", cP1);
    sprintf_s(TrackbarName, "CIRCLE_PARAM_2", cP2);
    sprintf_s(TrackbarName, "CIRCLE_RADIUS", maxHoughRadius);
    sprintf_s(TrackbarName, "FOV", 100);
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH), 
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->      
    createTrackbar(min1, trackbarWindowName, minInt1, 256, on_trackbar);
    createTrackbar(max1, trackbarWindowName, maxInt1, 256, on_trackbar);
    createTrackbar(min2, trackbarWindowName, minInt2, 256, on_trackbar);
    createTrackbar(max2, trackbarWindowName, maxInt2, 256, on_trackbar);
    createTrackbar(min3, trackbarWindowName, minInt3, 256, on_trackbar);
    createTrackbar(max3, trackbarWindowName, maxInt3, 256, on_trackbar);
    createTrackbar("MINAREA", trackbarWindowName, &MINAREA, 3000, on_trackbar );
    createTrackbar("MAXAREA", trackbarWindowName, &MAXAREA, 50000, on_trackbar );
    createTrackbar("ERODE", trackbarWindowName, &ERODE, 20, on_trackbar);
    createTrackbar("DILATE", trackbarWindowName, &DILATE, 20, on_trackbar);
    createTrackbar("CIRCLE_PARAM_1", trackbarWindowName, &cP1, 500, on_trackbar);
    createTrackbar("CIRCLE_PARAM_2", trackbarWindowName, &cP2, 100, on_trackbar);
    createTrackbar("CIRCLE_RADIUS", trackbarWindowName, &maxHoughRadius, 250, on_trackbar);
    createTrackbar("FOV", trackbarWindowName, &FOV, 100, on_trackbar);
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
double radiusDiff(Object a, Object b){
    return abs(a.getRadius() - b.getRadius());
}

vector<Object> findObjects(Mat &frame){
    double posX;
    double posY;
    Moments mom;
    vector<Vec4i> hierarchy;
    vector< vector<Point> > contours;
    vector<Object> objects;
    
    findContours(frame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

    //Check if any filtered objects was found
    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        for (int index = 0; index >= 0; index = hierarchy[index][0]) {
            mom = moments((Mat)contours[index], 1);
            double moment10 = mom.m10;
            double moment01 = mom.m01;
            double area = mom.m00;

            if (area >= MINAREA && area <= MAXAREA) {
                posX = moment10 / area;
                posY = moment01 / area;

                if (posX >= 0 && posY >= 0) {
                    //A true object is found, Add this object to the object vector
                    Object o;
                    o.setXPos(posX);
                    o.setYPos(posY);
                    o.setArea(area);
                    o.setRadius(sqrt(area/PI));
                    objects.push_back(o);
                }
            }
        }
    }
    return objects;
}

bool checkRadiusRatio(Object &a, Object &b, double ratio)
{
    if(a.getRadius() > b.getRadius())
    {
        return a.getRadius()/b.getRadius() <= ratio;
    }
    return b.getRadius()/a.getRadius() <= ratio;
}

void matchObjects(vector<Object> &first, vector<Object> &second, vector<Object> &result)
{
    for(int i = 0; i < first.size(); ++i)
    {
        for(int j = 0; j < second.size(); ++j)
        {
            //Check if the radius and position of the circles and the filtered objects match
            if(first.at(i).getXPos() - first.at(i).getRadius()/2 <= second.at(j).getXPos() && second.at(j).getXPos() <= first.at(i).getXPos() + first.at(i).getRadius()/2 &&
                first.at(i).getYPos() - first.at(i).getRadius()/2 <= second.at(j).getYPos() && second.at(j).getYPos() <= first.at(i).getYPos() + first.at(i).getRadius()/2 &&
                checkRadiusRatio(first.at(i), second.at(j), 1.5))
            {
                //We have matching circles/filtered obejcts
                //set the prio for the filtered objects
                first.at(i).incPrio(10);//abs(2/distanceFUCK(first.at(i), second.at(j))));
                first.at(i).incPrio(10);//2/radiusDiff(first.at(i), second.at(j)));
                //inc that there is a match
                first.at(i).incMatches();
                second.at(j).incMatches();
                
            }
        }
        result.push_back(first.at(i));
    }
    for(int i = 0; i < second.size(); ++i)
    {
        if(second.at(i).getMatches() == 0)
        {
            result.push_back(second.at(i));
        }
    }
}

void printPrio(vector<Object> &objects, vector<Object> &circles, vector<Object> &hsv, vector<Object> &ycrcb, Mat &frame)
{
    for(int i = 0; i<circles.size(); ++i)
    {
        circle(frame, Point(circles.at(i).getXPos(), circles.at(i).getYPos()), circles.at(i).getRadius(), Scalar(255,255,0), 2);
    }
    /*for(int i = 0; i<hsv.size(); ++i)
    {
        circle(frame, Point(hsv.at(i).getXPos(), hsv.at(i).getYPos()), hsv.at(i).getRadius(), Scalar(255,0,255));
    }
    for(int i = 0; i<ycrcb.size(); ++i)
    {
        circle(frame, Point(ycrcb.at(i).getXPos(), ycrcb.at(i).getYPos()), ycrcb.at(i).getRadius(), Scalar(0,255,255));
    }*/
    int avgDistance = 0;
    for(int i = 0; i < objects.size(); ++i)
    {
        //Print out the prio on every circle (both circles and filtered)
        stringstream ss;
        ss << objects.at(i).getPrio();
        cv::putText(frame, ss.str(), cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()-40), 1, 1, cv::Scalar(0, 0, 170), 2);

        if(i < 3 && objects.size() >= 3)
        {
            avgDistance += objects.at(i).getZDist();
            int j = (i+1)%3;
            cv::line(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), cv::Point(objects.at(j).getXPos(), objects.at(j).getYPos()),cv::Scalar(0, 0, 255), 1);
            cv::circle(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), objects.at(i).getRadius(), cv::Scalar(0, 0, 255));
        }
    }
    if(objects.size() >= 3)
    {
        avgDistance /= 3;
        stringstream ss2;
        ss2 << avgDistance;
        cv::putText(frame, ss2.str(), cv::Point(50, 50), 1, 3, cv::Scalar(0, 255, 255), 3);
    }
    


    
}

void circleToObject(vector<Vec3f> &circles, vector<Object> &result)
{
    for(int i = 0; i < circles.size(); i++)
    {
        Object o;
        o.setXPos(circles[i][0]);
        o.setYPos(circles[i][1]);
        o.setArea(circles[i][2]*circles[i][2]*PI);
        o.setRadius(circles[i][2]);
        result.push_back(o);
    }
}

/** Calculates the distance in 3D space */
double distance3D(Object a, Object b)
{
    return sqrt(pow(a.getXDist() - b.getXDist(), 2.0) + pow(a.getYDist() - b.getYDist(), 2.0) + pow(a.getZDist() - b.getZDist(), 2.0));
}

void createPairsByDistance(vector<Object> &objects, vector< vector<int> > &neighborhood)
{
    double distance;
    for(int i = 0; i < objects.size(); ++i)
    {
        vector<int> neighbors;
        for(int j = 0; j < objects.size(); ++j)
        {
            distance = distance3D(objects.at(i), objects.at(j));
            if(MIN_DISTANCE_BETWEEN_CIRCLES <= distance && distance <= MAX_DISTANCE_BETWEEN_CIRCLES)
            {
                neighbors.push_back(j);
                //H�r ritas avst�ndslinjer ut
                //cv::line(frameColor, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), cv::Point(objects.at(j).getXPos(), objects.at(j).getYPos()),cv::Scalar(255, 255, 0), 3);
            }

        }
        neighborhood.push_back(neighbors);
    }
}


void matchTriangles(vector<Object> &objects, vector< vector<int> > &neighborhood)
{
    for(int resident = 0; resident < neighborhood.size(); ++resident)
    {
        for(int neighbor = 0; neighbor < neighborhood.at(resident).size(); ++neighbor)
        {
            if(std::find(neighborhood.at(neighbor).begin(), neighborhood.at(neighbor).end(), resident) != neighborhood.at(neighbor).end() &&
                !objects.at(resident).getChecked())
            {
                objects.at(resident).setChecked(true);
                objects.at(resident).incPrio(10);
            }
        }
        
    }
}



void trackObjects(Mat &thresholdYCrCb, Mat &thresholdHSV, Mat &gray, vector<Object> &trackedYCrCb, vector<Object> &trackedHSV, vector<Object> &tackedCircles) {
    //A temporary Mat of the black&white frames are needed, otherwise the originals will be destroyed
    Mat tempFindYCrCb, tempHoughYCrCb, tempFindHSV, tempHoughHSV;
    thresholdYCrCb.copyTo(tempFindYCrCb);
    thresholdYCrCb.copyTo(tempHoughYCrCb);
    thresholdHSV.copyTo(tempFindHSV);
    thresholdHSV.copyTo(tempHoughHSV);
    //A temporary list in which the detected circles will be until they are converted into a list of Object
    vector<Vec3f> circlesTemp;

    //Get objects from frame
    trackedYCrCb = findObjects(tempFindYCrCb);
    trackedHSV = findObjects(tempFindHSV);

    // Apply the Hough Transform to find the circles in the frame
    HoughCircles(gray, circlesTemp, CV_HOUGH_GRADIENT, 1, 80, cP1, cP2, 0, maxHoughRadius);

    //Convert the circles to objects and put them in the "circles" vector
    circleToObject(circlesTemp, tackedCircles);
}

/**Calculate the 3D position for all objects in the given list */
void calculate3DPosition(vector<Object> &objects, Mat &frame, int ballRadius, int FOV)
{
	for(int i=0; i<objects.size(); i++){
        
        objects.at(i).setXDist( (ballRadius / objects.at(i).getRadius()) * (objects.at(i).getXPos() - frame.cols/2) );
        objects.at(i).setYDist( (ballRadius / objects.at(i).getRadius()) * (objects.at(i).getYPos() - frame.rows/2) );
        objects.at(i).setZDist( ballRadius * frame.cols / (2*objects.at(i).getRadius()*tan(FOV * PI/360)) );

        //Print the distance
        stringstream ss;
        ss << objects.at(i).getZDist();
        cv::putText(frame, ss.str(), cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()-20), 2, 0.5, cv::Scalar(0,180,180), 2);
        //stringstream ss2;
        //ss2 << objects.at(i).getRadius();
        //cv::putText(frame, ss2.str(), cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()+40), 1, 1, cv::Scalar(0,255,0));

    }
}

/** Calculatess and prints a normal to the found objects (a coordinate system) */
void printNormal(vector<Object> &objects, Mat &frame, int ballRadius, int FOV)
{
	if(objects.size()>=3){
        double position[3][3];
        for(int i=0; i<3; i++){
            position[i][0] = (frame.cols - objects.at(i).getXPos()) * ballRadius/objects.at(i).getRadius();
            position[i][1] = (frame.rows - objects.at(i).getYPos()) * ballRadius/objects.at(i).getRadius();
            position[i][2] = ballRadius * frame.rows / (2*objects.at(i).getRadius()*tan(FOV * PI/360));
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
        double w = objects.at(0).getXPos();
        double h = objects.at(0).getYPos();

        //Print
        cv::line(frame, cv::Point(w, h),cv::Point(w + vect1[0], h + vect1[1]),cv::Scalar(255,0,0), 3);
        cv::line(frame, cv::Point(w, h),cv::Point(w + vect2[0], h + vect2[1]),cv::Scalar(0,255,0), 3);
        cv::line(frame, cv::Point(w, h),cv::Point(w + vect3[0], h + vect3[1]),cv::Scalar(0,0,255), 3);
    }
}

int main(int argc, char** argv)
{
    
	int c;
    int debugMode = 1;
    int camInput = 0;/*
    while ( (c = getopt(argc, argv, "c:d")) != -1) {
        switch (c) {
            case 'c':
            camInput = atoi(optarg);
            break;
            case 'd':
            debugMode = 1;
            break;
            case '?':
            break;
            default:
            printf ("?? getopt returned character code 0%o ??\n", c);
        }
    }
    if (optind < argc) {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        printf ("\n");
    }

    if (debugMode){
        cout << "debugMode is on\n";
    }*/
    VideoCapture cam(camInput);

    //Sleep(1000);


    if (!cam.isOpened()) {
        cout << "Error loading camera";
    }
    else {
        cout << "Camera loaded OK\n\n";
    }

    //Colored frames
    Mat frameColor, frameHSV, frameYCrCb;
    //Black and white frames
    Mat thresholdHSV, thresholdYCrCb;
    //Grayscale frames
    Mat gray;
	
	
	vector<Object> trackedYCrCb;
    vector<Object> trackedHSV;
	vector<Object> trackedCircles;
	
	vector<Object> both;
	vector<Object> bothTemp;
    
    vector< vector<int> > neighborhood;

    /** Microsoft webcam
        Bl�tt papper
        YCbCr f�rger och HSV f�rger
    //For YCbCr filtering
    Y_MIN = 0;
    Y_MAX = 256;
    Cr_MIN = 100;//142;
    Cr_MAX = 256;
    Cb_MIN = 0;
    Cb_MAX = 127;//109;
    //For HSV filtering
    H_MIN = 0;
    H_MAX = 90;
    S_MIN = 40;//91;
    S_MAX = 256;
    V_MIN = 0;
    V_MAX = 256;
    //Same for both color filters
    MINAREA = 300;
    MAXAREA = 10000;
    cP1 = 300;
    cP2 = 20;
    ERODE = 1;
    DILATE = 2; */

    /** Microsoft webcam
        Cerist papper
        YCbCr f�rger och HSV f�rger 
    //For YCbCr filtering
    Y_MIN = 0;
    Y_MAX = 256;
    Cr_MIN = 110;
    Cr_MAX = 256;
    Cb_MIN = 150;
    Cb_MAX = 256;
    //For HSV filtering
    H_MIN = 105;
    H_MAX = 256;
    S_MIN = 33;
    S_MAX = 256;
    V_MIN = 0;
    V_MAX = 256;
    //Same for both color filters
    MINAREA = 300;
    MAXAREA = 10000;
    cP1 = 300;
    cP2 = 20;
    ERODE = 1;
    DILATE = 1;
    */

    /** Microsoft webcam
        cerisa bollar
        YCbCr f�rger och HSV f�rger*/
    //For YCbCr filtering
    Y_MIN = 0;
    Y_MAX = 256;
    Cr_MIN = 98;
    Cr_MAX = 156;
    Cb_MIN = 144;
    Cb_MAX = 256;
    //For HSV filtering
    H_MIN = 109;
    H_MAX = 170;
    S_MIN = 25;
    S_MAX = 228;
    V_MIN = 0;
    V_MAX = 256;
    //Same for both color filters
    MINAREA = 300;
    MAXAREA = 15000;
    cP1 = 195;
    cP2 = 20;
    ERODE = 1;
    DILATE = 1; 
    
    //Create the trackbars for both colors
    createTrackbars(1, "Y_MIN", "Y_MAX", "Cr_MIN", "Cr_MAX", "Cb_MIN", "Cb_MAX", &Y_MIN, &Y_MAX, &Cr_MIN, &Cr_MAX, &Cb_MIN, &Cb_MAX);
    createTrackbars(2, "H_MIN", "H_MAX", "S_MIN", "S_MAX", "V_MIN", "V_MAX", &H_MIN, &H_MAX, &S_MIN, &S_MAX, &V_MIN, &V_MAX);

    //Sleep(1000);
    
    //Force the image to be at 1280x720 _IF_ the camera supports it, otherwise it will be the cameras maximum res
    CvCapture* capture = cvCreateCameraCapture(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 720);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1280);

    while (1) {
        //Clear from old loop
        both.clear();
        bothTemp.clear();
        trackedYCrCb.clear();
        trackedHSV.clear();
        trackedCircles.clear();
        neighborhood.clear();

        //read the frame from the camera
        cam.read(frameColor);

        if (!frameColor.empty()){

            // Blur the image a bit
            GaussianBlur(frameColor, frameColor, Size(3, 3), 0, 0);
        
            // Convert from RGB to YCrCb
            cvtColor(frameColor, frameYCrCb, COLOR_RGB2YCrCb);
            // Convert from RGB to HSV
            cvtColor(frameColor, frameHSV, COLOR_RGB2HSV);

            // Convert YCrCb to binary B&W
            inRange(frameYCrCb, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX), thresholdYCrCb);
            // Convert HSV to binary B&W
            inRange(frameHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), thresholdHSV);

            //Gaussian the black/white image for YCrCb
            GaussianBlur(thresholdYCrCb, thresholdYCrCb, Size(3, 3), 0, 0);
            //Gaussian the black/white image for HSV
            GaussianBlur(thresholdHSV, thresholdHSV, Size(3, 3), 0, 0);
            //Convert to grayscale
            cvtColor(frameColor, gray, CV_BGR2GRAY);
            //Gaussian the grey image
            GaussianBlur(gray, gray, Size(3, 3), 0, 0);


            //morphops the binary image
            morphOps(thresholdYCrCb);
            morphOps(thresholdHSV);
            try {

                // Tracking objects from the image   
                trackObjects(thresholdYCrCb, thresholdHSV, gray, trackedYCrCb, trackedHSV, trackedCircles);
                
				//Match the filtered circles from YCrCb and HSV with eachother and set the prio
				matchObjects(trackedYCrCb, trackedHSV, bothTemp);
				matchObjects(bothTemp, trackedCircles, both);
				
				//Set the distance for X, Y and Z for all the objects in the both vector
				calculate3DPosition(both, frameColor, ballRadius, FOV);
				
				//Create pairs by their distance
				createPairsByDistance(both, neighborhood);
				
				//Match the pairs into triangles
				matchTriangles(both, neighborhood);
				
				//Sort the "both" vector with the highest prio first
				std::sort(both.begin(), both.end(), sorting);
				
				//If in debug mode, print the prio to the screen
				if(debugMode >= 1)
				{
                    printPrio(both, trackedCircles, trackedHSV, trackedYCrCb, frameColor);
					printNormal(both, frameColor, ballRadius, FOV);
				}

                // Display image
                imshow("Image", frameColor);
                //imshow("YCrCb image", ycrcbFrame);
                imshow("Binary image YCrCb", thresholdYCrCb);
                imshow("Binary image HSV", thresholdHSV);
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



