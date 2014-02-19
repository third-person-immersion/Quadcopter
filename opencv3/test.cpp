// OpenCV webcam test.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <windows.h>
#include "Frame.h"
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include "Object.h"
#define PI 3.14159265

using namespace cv;
using namespace std;

int MAX_DISTANCE_BETWEEN_CIRCLES = 30;
int MIN_DISTANCE_BETWEEN_CIRCLES = 10;
int MINAREA = 2000;
int MAXAREA = 2000;
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

//Skillnad i höjd (y) mellan två punkter i centimeter (tre punkter i triangel används)
//double twoPointsYDiff = 8.66;
//double twoPointsXDiff = 10;
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
int FOV = 66;//73; <-- 73 är mer rätt, men bättre med 70 då 73 är för diagonalen

vector<Object> both;

Mat frameColor, frameHSV, frameYCrCb;

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
    createTrackbar("MAXAREA", trackbarWindowName, &MAXAREA, 10000, on_trackbar );
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

void printPrio()
{
    for(int i = 0; i < both.size(); ++i)
    {
        //Print out the prio on every circle (both circles and filtered)
        stringstream ss;
        ss << both.at(i).getPrio();
        cv::putText(frameColor, ss.str(), cv::Point(both.at(i).getXPos(), both.at(i).getYPos()-40), 1, 1, cv::Scalar(0, 0, 170), 2);

        if(i < 3 && both.size() >= 3)
        {
            int j = (i+1)%3;
            cv::line(frameColor, cv::Point(both.at(i).getXPos(), both.at(i).getYPos()), cv::Point(both.at(j).getXPos(), both.at(j).getYPos()),cv::Scalar(0, 0, 255), 1);
            cv::circle(frameColor, cv::Point(both.at(i).getXPos(), both.at(i).getYPos()), both.at(i).getRadius(), cv::Scalar(0, 0, 255));
        }
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
        cv::circle(frameColor, cv::Point(circles[i][0], circles[i][1]), circles[i][2], cv::Scalar(255, 0, 90), 2);
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
                //Här ritas avståndslinjer ut
                //cv::line(frameColor, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), cv::Point(objects.at(j).getXPos(), objects.at(j).getYPos()),cv::Scalar(255, 255, 0), 3);
            }

        }
        neighborhood.push_back(neighbors);
    }
}


void matchTriangles(vector< vector<int> > &neighborhood)
{
    for(int resident = 0; resident < neighborhood.size(); ++resident)
    {
        for(int neighbor = 0; neighbor < neighborhood.at(resident).size(); ++neighbor)
        {
            if(std::find(neighborhood.at(neighbor).begin(), neighborhood.at(neighbor).end(), resident) != neighborhood.at(neighbor).end() &&
                !both.at(resident).getChecked())
            {
                both.at(resident).setChecked(true);
                both.at(resident).incPrio(10);
            }
        }
        
    }
}



void trackObjects(Mat &thresholdYCrCb, Mat &thresholdHSV, Mat &gray) {
    both.clear();
    
    Mat tempFindYCrCb, tempHoughYCrCb;
    thresholdYCrCb.copyTo(tempFindYCrCb);
    thresholdYCrCb.copyTo(tempHoughYCrCb);
    Mat tempFindHSV, tempHoughHSV;
    thresholdHSV.copyTo(tempFindHSV);
    thresholdHSV.copyTo(tempHoughHSV);

    //List of objects which came from circle detection
    vector<Object> circles, bothTemp;
    //List of circles from circle detection
    vector<Vec3f> circlesTemp;
    vector<Vec3f> circlesHoughYCrCb;
    vector<Vec3f> circlesHoughHSV;
    
    vector<Object> circlesYCrCb;
    vector<Object> circlesHSV;

    vector< vector<int> > neighborhood;

    

    //Get objects from frame
    circlesYCrCb = findObjects(tempFindYCrCb);
    circlesHSV = findObjects(tempFindHSV);

    
    // Apply the Hough Transform to find the circles in the frame
    //HoughCircles(tempHoughYCrCb, circlesHoughYCrCb, CV_HOUGH_GRADIENT, 1, 80, cP1, cP2, 0, maxHoughRadius);
    //HoughCircles(tempHoughYCrCb, circlesHoughHSV,   CV_HOUGH_GRADIENT, 1, 80, cP1, cP2, 0, maxHoughRadius);
    HoughCircles(gray,           circlesTemp,       CV_HOUGH_GRADIENT, 1, 80, cP1, cP2, 0, maxHoughRadius);


    //cout << "Found circles: " << circlesTemp.size() << "\n";

        
    //Convert the circles to objects and put them in the "circles" vector
    circleToObject(circlesTemp, circles);
    
    matchObjects(circlesYCrCb, circlesHSV, bothTemp);
    matchObjects(bothTemp, circles, both);

    cout << "\n\nNumber of hits both: " << both.size() << "\n\n";
    //Calculate the 3D position for all objects
    for(int i=0; i<both.size(); i++){
        
        both.at(i).setXDist( (ballRadius / both.at(i).getRadius()) * (both.at(i).getXPos() - frameColor.cols/2) );
        both.at(i).setYDist( (ballRadius / both.at(i).getRadius()) * (both.at(i).getYPos() - frameColor.rows/2) );
        both.at(i).setZDist( ballRadius * frameColor.cols / (2*both.at(i).getRadius()*tan(FOV * PI/360)) );

        //Print the distance
        stringstream ss;
        ss << both.at(i).getZDist();
        cv::putText(frameColor, ss.str(), cv::Point(both.at(i).getXPos(), both.at(i).getYPos()-20), 2, 0.5, cv::Scalar(0,180,180), 2);
        //stringstream ss2;
        //ss2 << both.at(i).getRadius();
        //cv::putText(frameColor, ss2.str(), cv::Point(both.at(i).getXPos(), both.at(i).getYPos()+40), 1, 1, cv::Scalar(0,255,0));

    }

    //Create pairs
    createPairsByDistance(both, neighborhood);
    
    matchTriangles(neighborhood);

    
    //Sort the "both" vector with the highest prio first
    std::sort(both.begin(), both.end(), sorting);

    
    printPrio();

    //Allt detta är för koordinatsystemet. Ja.
    if(both.size()>=3){
        double position[3][3];
        for(int i=0; i<3; i++){
            position[i][0] = (frameColor.cols - both.at(i).getXPos()) * ballRadius/both.at(i).getRadius();
            position[i][1] = (frameColor.rows - both.at(i).getYPos()) * ballRadius/both.at(i).getRadius();
            position[i][2] = ballRadius * frameColor.rows / (2*both.at(i).getRadius()*tan(FOV * PI/360));
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
        cv::line(frameColor, cv::Point(w, h),cv::Point(w + vect1[0], h + vect1[1]),cv::Scalar(255,0,0), 3);
        cv::line(frameColor, cv::Point(w, h),cv::Point(w + vect2[0], h + vect2[1]),cv::Scalar(0,255,0), 3);
        cv::line(frameColor, cv::Point(w, h),cv::Point(w + vect3[0], h + vect3[1]),cv::Scalar(0,0,255), 3);
    }

}





int main(int argc, char** argv)
{
	int c;
    int debugMode = 0;
    int camInput = 0;
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
    }
    VideoCapture cam(camInput);
    
    //Sleep(1000);


    if (!cam.isOpened()) {
        cout << "Error loading camera";
    }
    else {
        cout << "Camera loaded OK\n\n";
    }

    //svartvita bilden (den filtrerade) för hsv
    Mat thresholdHSV;
    Mat thresholdYCrCb;
    //grayscale image
    Mat gray;

    /** Microsoft webcam
        Blått papper
        YCbCr färger och HSV färger
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
        YCbCr färger och HSV färger */
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


    /** Microsoft webcam
        Tennisbollar
        YCbCr färger och HSV färger
    //For YCbCr filtering
    Y_MIN = 0;
    Y_MAX = 256;
    Cr_MIN = 68;
    Cr_MAX = 124;
    Cb_MIN = 132;
    Cb_MAX = 226;
    //For HSV filtering
    H_MIN = 58;
    H_MAX = 256;
    S_MIN = 74;
    S_MAX = 195;
    V_MIN = 0;
    V_MAX = 256;
    //Same for both color filters
    MINAREA = 300;
    MAXAREA = 1500;
    cP1 = 195;
    cP2 = 20;
    ERODE = 1;
    DILATE = 1; */
    
    //Create the trackbars for both colors
    createTrackbars(1, "Y_MIN", "Y_MAX", "Cr_MIN", "Cr_MAX", "Cb_MIN", "Cb_MAX", &Y_MIN, &Y_MAX, &Cr_MIN, &Cr_MAX, &Cb_MIN, &Cb_MAX);
    createTrackbars(2, "H_MIN", "H_MAX", "S_MIN", "S_MAX", "V_MIN", "V_MAX", &H_MIN, &H_MAX, &S_MIN, &S_MAX, &V_MIN, &V_MAX);

    //Sleep(1000);
    
    //Force the image to be at 1280x720 _IF_ the camera supports it, otherwise it will be the cameras maximum res
    CvCapture* capture = cvCreateCameraCapture(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 720);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1280);

    while (1) {
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

                // Tracking    
                trackObjects(thresholdYCrCb, thresholdHSV, gray);
                

                //Canny(gray, gray, cP1/3, cP1);
                
                

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



