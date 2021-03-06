﻿#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <stdio.h>
#include <thread>
#include <iterator>
#include <iostream>

#include "tclap/CmdLine.h"
#include <sys/timeb.h>

#define PI 3.14159265
#include "Object.h"

#ifdef _WIN32
#include "../cam-share/Read_windows.h"
#else
#include "../cam-share/Read_unix.h"
#endif



using namespace TCLAP;
using namespace cv;
using namespace std;


//Delimiters
char DELIMITER_GROUP = 29;
char DELIMITER_RECORD = 30;

int MAX_DISTANCE_BETWEEN_CIRCLES = 52;
int MIN_DISTANCE_BETWEEN_CIRCLES = 26; //40
int MINAREA = 2000;
int MAXAREA = 15000;
int ERODE = 30;
int DILATE = 60;
int Y_MIN = 0;
int Y_MAX = 256;
int Cb_MIN = 81;
int Cb_MAX = 224;
int Cr_MIN = 147;
int Cr_MAX = 256;
int H_MIN = 116;
int H_MAX = 176;
int S_MIN = 77;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
int cP1 = 400;
int cP2 = 30; //cerist papper
int maxHoughRadius = 80;
int darkenFactor = 1;


//The width of the sensor on the camera (in cm)
int sensorWidth = 6.248 * 0.1; //5.75
//The focal length (in cm)
int focalLength = 16.53 * 0.1; //14

//The radius of the current ball used
double ballRadius = 7.5/2;
//The Field Of View of the camera used at the moment
//Microsoft webcam
int FOV_H = 62;//;123;//48;//66; //48;
int FOV_V = 94;

int doWeHaveNiceValues = 0;

string windowTitle = "frame";

// FPS - Get count in millisoconds
int getMilliCount(){
    timeb tb;
    ftime(&tb);
    int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
    return nCount;
}

// FPS - Calculate the difference between start and end time.
int getMilliSpan(int timeStart){
    int span = getMilliCount() - timeStart;
    if (span < 0)
        span += 0x100000 * 1000;
    return span;
}

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
    cv::namedWindow(trackbarWindowName, 0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
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
    createTrackbar("FOV_H", trackbarWindowName, &FOV_H, 140, on_trackbar);
}

void morphOps(Mat &thresh){

    //create structuring element that will be used to "dilate" and "erode" image.
    //the element chosen here is a rectangle with size from variables ERODE/DILATE

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

double pix2Dist(double pix, double objRadius, double ballRadius)
{
    return pix * ballRadius/objRadius;
}
double dist2Pix(double dist, double objDistanceInPix, double ballDistanceInCM)
{
    return dist * objDistanceInPix/ballDistanceInCM;
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

void findObjects(Mat &frame, vector<Object> &objects){
    double posX;
    double posY;
    Moments mom;
    vector<Vec4i> hierarchy;
    vector< vector<Point> > contours;

    findContours(frame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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
}

/** Checks the ratio between object a and b. If it is less than or equal to the given ratio, True will be returned  */
bool checkRadiusRatio(Object &a, Object &b, double ratio)
{
    if(a.getRadius() > b.getRadius())
    {
        return a.getRadius()/b.getRadius() <= ratio;
    }
    return b.getRadius()/a.getRadius() <= ratio;
}

void matchObjects(vector<Object> &first, vector<Object> &second, vector<Object> &result, bool radiusCheck)
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

                if(radiusCheck)
                {
                    double newRadius = (first.at(i).getRadius()*2 + second.at(j).getRadius())/3;
                    double newXPos = (first.at(i).getXPos()*2 + second.at(j).getXPos())/3;
                    double newYPos = (first.at(i).getYPos()*2 + second.at(j).getYPos())/3;
                    first.at(i).setRadius(newRadius);
                    second.at(j).setRadius(newRadius);
                    first.at(i).setXPos(newXPos);
                    second.at(j).setXPos(newXPos);
                    first.at(i).setYPos(newYPos);
                    second.at(j).setYPos(newYPos);
                }
                else
                {
                    double newRadius = (first.at(i).getRadius() + second.at(j).getRadius())/2;
                    double newXPos = (first.at(i).getXPos() + second.at(j).getXPos())/2;
                    double newYPos = (first.at(i).getYPos() + second.at(j).getYPos())/2;
                    first.at(i).setRadius(newRadius);
                    second.at(j).setRadius(newRadius);
                    first.at(i).setXPos(newXPos);
                    second.at(j).setXPos(newXPos);
                    first.at(i).setYPos(newYPos);
                    second.at(j).setYPos(newYPos);
                }


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

void printPrio(vector<Object> &objects, Mat &frame)
{
    //int avgDistance = 0;
    for(int i = 0; i < objects.size(); ++i)
    {
        //Print out the prio on every circle (both circles and filtered)
        stringstream ss;
        ss << objects.at(i).getPrio();
        cv::putText(frame, ss.str(), cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()-40), 1, 1, cv::Scalar(0, 0, 170), 2);
        //cv::putText(frame, "Z: " + std::to_string(objects.at(i).getZDist()), cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()-20), 1, 1, cv::Scalar(0, 0, 255));

        if(i < 3 && objects.size() >= 3)
        {
            //avgDistance += objects.at(i).getZDist();
            int j = (i+1)%3;
            cv::line(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), cv::Point(objects.at(j).getXPos(), objects.at(j).getYPos()),cv::Scalar(0, 0, 255), 1);
            //cv::circle(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), objects.at(i).getRadius(), cv::Scalar(0, 0, 255));
        }
    }
    /*
    if(objects.size() >= 3)
    {
        avgDistance /= 3;
        stringstream ss2;
        ss2 << avgDistance << " cm :D";
        cv::putText(frame, ss2.str(), cv::Point(50, 50), 1, 3, cv::Scalar(0, 255, 255), 3);
    }
    */
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

void createPairsByDistance(vector<Object> &objects, vector< vector<int> > &neighborhood, Mat &frame)
{
    double distance;

    int xPos, yPos;
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
                cv::line(frame, cv::Point(objects.at(i).getXPos(), objects.at(i).getYPos()), cv::Point(objects.at(j).getXPos(), objects.at(j).getYPos()),cv::Scalar(255, 255, 255), 2);

                if(objects.at(i).getXPos() <= objects.at(j).getXPos())
                {
                    xPos = objects.at(i).getXPos() + (objects.at(j).getXPos() - objects.at(i).getXPos())/2;
                }
                else 
                {
                    xPos = objects.at(j).getXPos() + (objects.at(i).getXPos() - objects.at(j).getXPos())/2;
                }
                if(objects.at(i).getYPos() <= objects.at(j).getYPos())
                {
                    yPos = objects.at(i).getYPos() + (objects.at(j).getYPos() - objects.at(i).getYPos())/2;
                }
                else 
                {
                    yPos = objects.at(j).getYPos() + (objects.at(i).getYPos() - objects.at(j).getYPos())/2;
                }

                putText(frame, std::to_string(distance),Point(xPos, yPos), 1, 1.2, cv::Scalar(255, 255, 255), 1);
            }

        }
        neighborhood.push_back(neighbors);
    }
}


void matchTriangles(vector<Object> &objects, vector< vector<int> > &neighborhood)
{
    double maxPrio;
    for(int resident = 0; resident < neighborhood.size(); ++resident)
    {
        maxPrio = objects.at(resident).getPrio();
        for(int neighborIndex = 0; neighborIndex < neighborhood.at(resident).size(); ++neighborIndex)
        {
            int neighbor = neighborhood.at(resident).at(neighborIndex);
            if(neighbor == resident)
            {
                continue;
            }
            if(maxPrio < objects.at(neighbor).getPrio())
            {
                maxPrio = objects.at(neighbor).getPrio();
            }
            for(int neighborsNeighborIndex = 0; neighborsNeighborIndex < neighborhood.at(neighbor).size(); ++neighborsNeighborIndex)
            {
                int neighborsNeighbor = neighborhood.at(neighbor).at(neighborsNeighborIndex);
                if(neighborsNeighbor == neighbor)
                {
                    continue;
                }
                if(maxPrio < objects.at(neighborsNeighbor).getPrio())
                {
                    maxPrio = objects.at(neighborsNeighbor).getPrio();
                }
                if(std::find(neighborhood.at(neighborsNeighbor).begin(), neighborhood.at(neighborsNeighbor).end(), resident) != neighborhood.at(neighborsNeighbor).end() &&
                    !objects.at(resident).getIncludedInTriangle())
                {
                    //Här har vi hittat trianglar!!
                    objects.at(resident).setIncludedInTriangle(true);
                    objects.at(resident).incPrio(20);
                    objects.at(resident).incPrio((maxPrio - objects.at(resident).getPrio())/2);
                }
            }
        }

    }
}



void trackCircles(Mat &frame, Mat& gray, vector<Object> &tackedCircles, bool convertToGray) {
    //Convert to grayscale
    if(convertToGray)
    {
        cvtColor(frame, gray, CV_BGR2GRAY);
    }

    //Gaussian the grey image
    //GaussianBlur(gray, gray, Size(3, 3), 0, 0);
    //A temporary list in which the detected circles will be until they are converted into a list of Object
    vector<Vec3f> circlesTemp;
    // Apply the Hough Transform to find the circles in the frame
    HoughCircles(gray, circlesTemp, CV_HOUGH_GRADIENT, 1, 80, cP1, cP2, 0, maxHoughRadius);
    //Convert the circles to objects and put them in the "circles" vector
    circleToObject(circlesTemp, tackedCircles);
}

/**Calculate the 3D position for all objects in the given list */
void calculate3DPosition(vector<Object> &objects, Mat &frame, double ballRadius, int FOV_H, int FOV_V)
{
    for(int i=0; i<objects.size(); i++){
        objects.at(i).setXDist( (ballRadius / objects.at(i).getRadius()) * (objects.at(i).getXPos() - frame.cols/2) );
        objects.at(i).setYDist( (ballRadius / objects.at(i).getRadius()) * (objects.at(i).getYPos() - frame.rows/2) );
        objects.at(i).setZDist( ballRadius * frame.cols / (2*objects.at(i).getRadius()*tan(FOV_H * PI/360) ) );
    }
}

void calculate3DPositionUsingFocalLength(vector<Object> &objects, Mat &frame, double ballRadius, int FOV_H, int FOV_V)
{
    for(int i=0; i<objects.size(); i++){
        objects.at(i).setXDist( (ballRadius / objects.at(i).getRadius()) * (objects.at(i).getXPos() - frame.cols/2) );
        objects.at(i).setYDist( (ballRadius / objects.at(i).getRadius()) * (objects.at(i).getYPos() - frame.rows/2) );
        objects.at(i).setZDist( (ballRadius*frame.cols*focalLength)/(objects.at(i).getRadius()*sensorWidth) );
    }
}

/** Calculates the normal and mid vector. If dFlag is >= 2, then it prints aswell.
    The given Object list must be sorted! */
void calculatePlane(vector<Object> &objects, vector<double> &midPos, vector<double> &angles, Mat &frame, double ballRadius, int FOV, int dFlag)
{
    if(objects.size()>=3)
    {
        vector<double> vect1;
        vector<double> vect2;
        vector<double> normal;
        double lineLength = 10;

        //Calculate the first vector between the first and second object
        vect1.push_back(objects.at(1).getXDist() - objects.at(0).getXDist());
        vect1.push_back(objects.at(1).getYDist() - objects.at(0).getYDist());
        vect1.push_back(objects.at(1).getZDist() - objects.at(0).getZDist());

        if(dFlag >= 3)
        {
            cv::line(
                //Which frame to draw in
                frame,
                //Start pos
                cv::Point(objects.at(0).getXPos(), objects.at(0).getYPos()),
                //End pos
                cv::Point(
                    objects.at(0).getXPos() + dist2Pix(vect1.at(0), objects.at(0).getRadius(), ballRadius),
                    objects.at(0).getYPos() + dist2Pix(vect1.at(1), objects.at(0).getRadius(), ballRadius)),
                //Color
                cv::Scalar(255, 0, 0),
                //Scale
                3);
        }

        //Set the initial values for the mid position
        midPos.push_back(objects.at(0).getXDist() + vect1.at(0)/3);
        midPos.push_back(objects.at(0).getYDist() + vect1.at(1)/3);
        midPos.push_back(objects.at(0).getZDist() + vect1.at(2)/3);

        if(dFlag >= 3)
        {
            putText(frame, "vect1 Z: " + std::to_string(vect1.at(2)) + " cm",Point(50,220), 1, 1.2, cv::Scalar(0, 255, 255), 1);
        }




        //Calculate the second vector between the first and third object
        vect2.push_back(objects.at(2).getXDist() - objects.at(0).getXDist());
        vect2.push_back(objects.at(2).getYDist() - objects.at(0).getYDist());
        vect2.push_back(objects.at(2).getZDist() - objects.at(0).getZDist());


        if(dFlag >= 3)
        {
            cv::line(
                //Which frame to draw in
                frame,
                //Start pos
                cv::Point(objects.at(0).getXPos(), objects.at(0).getYPos()),
                //End pos
                cv::Point(
                    //X position
                    objects.at(0).getXPos() + dist2Pix(vect2.at(0), objects.at(0).getRadius(), ballRadius),
                    //Y position
                    objects.at(0).getYPos() + dist2Pix(vect2.at(1), objects.at(0).getRadius(), ballRadius)),
                //Color
                cv::Scalar(0,255,0),
                //Scale
                3);
        }

        //Add the values from the second vector to the mid position
        midPos.at(0) += vect2.at(0)/3;
        midPos.at(1) += vect2.at(1)/3;
        midPos.at(2) += vect2.at(2)/3;

        if(dFlag >= 3)
        {
            putText(frame, "vect2 Z: " + std::to_string(vect2.at(2)) + " cm",Point(50,240), 1, 1.2, cv::Scalar(0, 255, 255), 1);
        }
        if(dFlag >= 1)
        {
            circle(frame, Point(frame.cols/2 + dist2Pix(midPos.at(0), objects.at(0).getRadius(), ballRadius), frame.rows/2 + dist2Pix(midPos.at(1), objects.at(0).getRadius(), ballRadius)), 3, cv::Scalar(0, 255, 255), 3);
        }



        //Calculates the normal vector using the first and second vector (see http://en.wikipedia.org/wiki/Cross_product for calculations)
        double norX = vect1.at(1)*vect2.at(2) - vect1.at(2)*vect2.at(1);
        double norY = vect1.at(2)*vect2.at(0) - vect1.at(0)*vect2.at(2);
        double norZ = vect1.at(0)*vect2.at(1) - vect1.at(1)*vect2.at(0);

        //If the Z direction is negative, make it positive since we want the normal vector to point towards the camera at all times
        if(norZ > 0)
        {
            norX *= -1;
            norY *= -1;
            norZ *= -1;
        }

        //Create the normal vector
        normal.push_back(norX);
        normal.push_back(norY);
        normal.push_back(norZ);

        //Normalize the first vector length
        double length = sqrt(pow(vect1.at(0), 2) + pow(vect1.at(1), 2) + pow(vect1.at(2), 2));
        for(int i=0; i<3; i++){
            vect1.at(i) = vect1.at(i)/length;
        }

        //Normalize the second vector length
        length = sqrt(pow(vect2.at(0), 2) + pow(vect2.at(1), 2) + pow(vect2.at(2), 2));
        for(int i=0; i<3; i++){
            vect2.at(i) = vect2.at(i)/length;
        }

        //Normalize the normal vector length
        length = sqrt(pow(normal.at(0), 2) + pow(normal.at(1), 2) + pow(normal.at(2), 2));
        for(int i=0; i<3; i++){
            normal.at(i) = normal.at(i)/length;
        }

        if(dFlag >= 3)
        {
            putText(frame, "normal X: " + std::to_string(normal.at(0)) + "cm",Point(50,160), 1, 1.2, cv::Scalar(0, 255, 255), 1);
            putText(frame, "normal Y: " + std::to_string(normal.at(1)) + "cm",Point(50,180), 1, 1.2, cv::Scalar(0, 255, 255), 1);
            putText(frame, "normal Z: " + std::to_string(normal.at(2)) + "cm",Point(50,200), 1, 1.2, cv::Scalar(0, 255, 255), 1);
        }

        //Calculate the start positions for the coordinate system lines
        double startX = frame.cols/2 + dist2Pix(midPos.at(0), objects.at(0).getRadius(), ballRadius);
        double startY = frame.rows/2 + dist2Pix(midPos.at(1), objects.at(0).getRadius(), ballRadius);

        //Calculate and create the angle vector
        angles.push_back((180/PI) * atan(normal.at(0)/normal.at(2)));
        angles.push_back((180/PI) * atan(normal.at(1)/normal.at(2)));
        angles.push_back((180/PI) * atan(normal.at(0)/normal.at(1)));

        if(dFlag >=2)
        {
            //Print Vect1
            cv::line(frame,
                //Start pos (Is the mid pos vector position)
                cv::Point(startX, startY),
                //End pos
                cv::Point(
                    startX + lineLength*dist2Pix(vect1.at(0), objects.at(0).getRadius(), ballRadius),
                    startY + lineLength*dist2Pix(vect1.at(1), objects.at(0).getRadius(), ballRadius)),
                //Color
                cv::Scalar(255,0,0),
                //Scale
                3);

            /*
            //Start pos
            cv::Point(objects.at(0).getXPos(), objects.at(0).getYPos()),
            //End pos
            cv::Point(
                //X position
                objects.at(0).getXPos() + dist2Pix(vect2.at(0), objects.at(0).getRadius(), ballRadius),
                //Y position
                objects.at(0).getYPos() + dist2Pix(vect2.at(1), objects.at(0).getRadius(), ballRadius)),
                */

            //Print Vect2
            cv::line(frame,
                //Start pos (Is the mid pos vector position)
                cv::Point(startX, startY),
                //End pos
                cv::Point(
                    startX + lineLength*dist2Pix(vect2.at(0), objects.at(0).getRadius(), ballRadius),
                    startY + lineLength*dist2Pix(vect2.at(1), objects.at(0).getRadius(), ballRadius)),
                //Color
                cv::Scalar(0,255,0),
                //Scale
                3);

            //Print Normal
            cv::line(frame,
                //Start pos (Is the mid pos vector position)
                cv::Point(startX, startY),
                //End pos
                cv::Point(
                    startX + lineLength*dist2Pix(normal.at(0), objects.at(0).getRadius(), ballRadius),
                    startY + lineLength*dist2Pix(normal.at(1), objects.at(0).getRadius(), ballRadius)),
                //Color
                cv::Scalar(0,0,255),
                //Scale
                3);
        }

    }
}

void matchLast(vector<Object> &last, vector<Object> &dst, int index)
{
    for(int i = 0; i < dst.size(); ++i)
    {
        for(int j = 0; j < last.size(); ++j)
        {
            //Check if the radius and position of the circles and the filtered objects match
            if(dst.at(i).getXPos() - dst.at(i).getRadius()*(5 - index)/2 <= last.at(j).getXPos() && last.at(j).getXPos() <= dst.at(i).getXPos() + dst.at(i).getRadius()*(5 - index)/2 &&
                dst.at(i).getYPos() - dst.at(i).getRadius()*(5 - index)/2 <= last.at(j).getYPos() && last.at(j).getYPos() <= dst.at(i).getYPos() + dst.at(i).getRadius()*(5 - index)/2 &&
                checkRadiusRatio(dst.at(i), last.at(j), (2.5 - 0.25*index)))
            {
                dst.at(i).incPrio(4*(index + 1));
            }
        }
    }
}

void copyObject(Object &src, vector<Object> &dst)
{
    Object o;
    o.setXPos(src.getXPos());
    o.setYPos(src.getYPos());
    o.setArea(src.getArea());
    o.setRadius(src.getRadius());
    o.setAdded(src.getAdded());
    o.setIncludedInTriangle(src.getIncludedInTriangle());
    o.setPrio(src.getPrio());
    o.setType(src.getType());
    o.setXDist(src.getXDist());
    o.setYDist(src.getYDist());
    o.setZDist(src.getZDist());
    dst.push_back(o);
}

void trackHSVObjects(Mat &frame, Mat &threashold, vector<Object> &foundObjects)
{
    Mat target;
    //Black and white frames
    Mat frameThreshold;
    threashold.copyTo(frameThreshold);

    // Convert from frame (RGB) to HSV
    cvtColor(frame, target, COLOR_RGB2HSV);

    // Convert traget to binary B&W
    inRange(target, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), frameThreshold);

    //Gaussian the black/white image
    GaussianBlur(frameThreshold, frameThreshold, Size(3, 3), 0, 0);

    //morphops the black/white image
    morphOps(frameThreshold);

    findObjects(frameThreshold, foundObjects);
}

void trackYCrCbObjects(Mat &frame, Mat &threashold, vector<Object> &foundObjects)
{
    Mat target;
    //Black and white frames
    Mat frameThreshold;
    threashold.copyTo(frameThreshold);

    // Convert from frame (RGB) to YCrCb
    cvtColor(frame, target, COLOR_RGB2YCrCb);

    // Convert traget to binary B&W
    inRange(target, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX), frameThreshold);

    //Gaussian the black/white image
    GaussianBlur(frameThreshold, frameThreshold, Size(3, 3), 0, 0);

    //morphops the black/white image
    morphOps(frameThreshold);

    findObjects(frameThreshold, foundObjects);
}

void trackObjects(Mat &frame, Mat &threashold, vector<Object> &foundObjects, int code, Scalar scalarMin, Scalar scalarMax)
{
    Mat target;
    //Black and white frames
    Mat frameThreshold;

    // Convert from frame (RGB) to YCrCb
    cvtColor(frame, target, code);

    //Gaussian the filtered image
    GaussianBlur(target, target, Size(9, 9), 0, 0);

    // Convert traget to binary B&W
    inRange(target, scalarMin, scalarMax, threashold);

    //Gaussian the black/white image
    GaussianBlur(threashold, threashold, Size(3, 3), 0, 0);

    //morphops the black/white image
    morphOps(threashold);

    threashold.copyTo(frameThreshold);

    findObjects(frameThreshold, foundObjects);
}

void darkenMatrix(Mat &frame)
{
    frame.convertTo(frame, CV_32FC3);
    for(int y=0; y<frame.rows; y++)
        for(int x=0; x<frame.cols; x++)
            for(int c=0;c<3;c++)
            frame.at<Vec3f>(y,x)[c] = (darkenFactor/10.0)*pow(frame.at<Vec3f>(y,x)[c]/255.0,3);
}

int main(int argc, char** argv)
{
    string vflag = "";
    string fflag = "";
    int dflag = 0;
    int rflag = 0;
    int distflag = 0;
    try
    {
        string desc = "This is the command description";
        CmdLine cmd(desc, ' ', "0.1");
        ValueArg<int> debug ("d", "debug", "Activate debug mode", false, 0, "int");
        ValueArg<string> video ("v", "video", "Save video", false, "", "string");
        ValueArg<string> file ("f", "logfile", "Set log file path", false, "caminfo.log", "string");
        ValueArg<int> release ("r", "release", "Activate release mode", false, 0, "int");
        ValueArg<int> distance ("s", "distance", "Set the desired distance", false, 0, "int");
        cmd.add( debug );
        cmd.add( video );
        cmd.add( file );
        cmd.add( release );
        cmd.add( distance );
        // Parse arguments
        cmd.parse( argc, argv );

        // Do what you intend too...
        dflag = debug.getValue();
        vflag = video.getValue();
        fflag = file.getValue();
        rflag = release.getValue();
        distflag = distance.getValue();
    }
    catch ( ArgException& e )
    {
        cout << "ERROR: " << e.error() << " " << e.argId() << endl;
        return 1;
    }

    if(dflag >= 1 && rflag <= 0)
    {
        cout << "Logfile set to: " << fflag << endl;
    }

    //Create a read object which will access the shared memory
    Read *pRead = new Read(fflag);
    //Set the size which will be used then recoding the video
    cv::Size camSize(pRead->getWidth(), pRead->getHeight());

    //Varibles for FPS counting
    int startTime;
    int endTime;
    double fps;
    bool printFPS = false;
    bool writeVideo = true;
    bool loop = true;

    //Colored frames
    Mat frameColor, frameGray, threasholdYCrCb, threasholdHSV, frameColorDark, frameGrayThresholdYCrCb, frameGrayThresholdHSV, frameColorUntouched;


    vector<Object> trackedYCrCb;
    vector<Object> trackedHSV;
    vector<Object> trackedCircles;
    vector<Object> trackedCirclesYCrCb;
    vector<Object> trackedCirclesHSV;
    vector<Object> both;
    vector<Object> bothTemp;
    list< vector<Object> > lastPrios;
    vector< vector<int> > neighborhood;
    vector<double> angleBuff;
    vector<double> distanceBuff;

    vector<Object> tempVector;

    /********************************************************************\
    **    HERE IS THE ACTUAL DATA WE NEED TO POSITION THE QUADCOPTER    **
    \********************************************************************/

    //The acctual interesting schtuff
    vector<double> midPos;
    //The angles to the plane;
    vector<double> angles;

    //Value that says how sure we are on our objects we found. Note that it will
    //rarly reach 100% (1.0), but be around 50-70 in bad lighting (not tested in good lighting)
    double sure;

    /** Microsoft webcam
        cerisa bollar
        YCbCr färger och HSV färger*/

    //For YCbCr filtering
    Y_MIN = 0;
    Y_MAX = 256;
    Cr_MIN = 114;//109;
    Cr_MAX = 158;//165;
    Cb_MIN = 151;//156;
    Cb_MAX = 230;//242;
    //For HSV filtering
    H_MIN = 117;//112;
    H_MAX = 142;//133;
    S_MIN = 119;//88;
    S_MAX = 256;//256;
    V_MIN = 0;
    V_MAX = 256;
    //Same for both color filters
    MINAREA = 100;
    MAXAREA = 30000;
    cP1 = 90;
    cP2 = 12;
    maxHoughRadius = 30;
    ERODE = 1;
    DILATE = 1;

    string windowYCrCb = "YCrCb image";
    string windowHSV = "HSV image";
    string windowGray = "Gray image";
    string windowYCrCb2 = "YCrCb image 2";
    string windowHSV2 = "HSV image 2";

    if(dflag >= 1)
    {
        if(rflag <= 0)
        {
            cout << "Debug mode is ON and set to: " << dflag << "\n";
            cout << "Press F to see FPS in console. Press Q to quit.\n";
        }

        if(dflag >= 2)
        {
            //Create the trackbars for both colors
            createTrackbars(1, "Y_MIN", "Y_MAX", "Cr_MIN", "Cr_MAX", "Cb_MIN", "Cb_MAX", &Y_MIN, &Y_MAX, &Cr_MIN, &Cr_MAX, &Cb_MIN, &Cb_MAX);
            createTrackbars(2, "H_MIN", "H_MAX", "S_MIN", "S_MAX", "V_MIN", "V_MAX", &H_MIN, &H_MAX, &S_MIN, &S_MAX, &V_MIN, &V_MAX);

            if(rflag <= 0)
            {
                cout << "Starting to capture!\nBall radius set to: " << ballRadius << "\n";
            }


            cv::namedWindow(windowYCrCb, CV_WINDOW_AUTOSIZE );
            cv::namedWindow(windowHSV, CV_WINDOW_AUTOSIZE );
            cv::namedWindow(windowGray, CV_WINDOW_AUTOSIZE );

            //cv::namedWindow(windowYCrCb2, CV_WINDOW_AUTOSIZE );
            //cv::namedWindow(windowHSV2, CV_WINDOW_AUTOSIZE );
        }

        //Create window
        cv::namedWindow(windowTitle, CV_WINDOW_FREERATIO );

        // Start timer for fps counting
        startTime = getMilliCount();
    }

    //creat output for video saving
    cv::VideoWriter outputColor, outputHSV, outputYCrCb, outputGray;
    if(!vflag.empty()){
        if(dflag >= 1 && rflag <= 0)
        {
            cout << "Videoflag set! filename: " << vflag << "\n";
        }

        outputColor.open ( vflag + "Color.avi", CV_FOURCC('D','I','V','X'), 15, camSize, true );
        outputHSV.open ( vflag + "HSV.avi", CV_FOURCC('D','I','V','X'), 15, camSize, false );
        outputYCrCb.open ( vflag + "YCrCb.avi", CV_FOURCC('D','I','V','X'), 15, camSize, false );
        outputGray.open ( vflag + "Gray.avi", CV_FOURCC('D','I','V','X'), 15, camSize, false );

        if (!outputColor.isOpened() && dflag >= 1 && rflag <= 0)
        {
            cout << "Output frameColor could not be opened\n";
        }
        if (!outputHSV.isOpened() && dflag >= 1 && rflag <= 0)
        {
            cout << "Output thresholdHSV could not be opened\n";
        }
        if (!outputYCrCb.isOpened() && dflag >= 1 && rflag <= 0)
        {
            cout << "Output thresholdYCrCb could not be opened\n";
        }
        if (!outputGray.isOpened() && dflag >= 1 && rflag <= 0)
        {
            cout << "Output gray could not be opened\n";
        }
    }

    vector<Mat> channels;

    //Force the image to be at 720x576 which is the frame-grabber's maximum resolution
   
    while (loop) {

        //Clear from old loop
        both.clear();
        bothTemp.clear();
        trackedYCrCb.clear();
        trackedHSV.clear();
        trackedCircles.clear();
        neighborhood.clear();
        midPos.clear();
        angles.clear();

        //tempVector.clear();

        //read the frame from the shared camera memory
        pRead->getFrame().copyTo(frameColor);

        frameColor.copyTo(frameColorUntouched);

        // FPS viewer
        if (dflag >= 1) {
            // FPS viewer
            double milliSpan = (double)getMilliSpan(startTime) / 1000;

            fps = 1 / milliSpan;
            if (printFPS && rflag <= 0) {
                cout << "FPS: " << fps << "\n";
            }
            startTime = getMilliCount();
        }


        if (!frameColor.empty()){
            // Blur the image a bit
            //GaussianBlur(frameColor, frameColor, Size(1, 1), 0, 0);

            //These can be used if threaded calculations is desired. Just dont forget to join (begining of try-block)

            /*
            //Start thread 1 that will handle the YCrCb color
            thread YCrCbThread(trackYCrCbObjects, std::ref(frameColor), std::ref(threasholdYCrCb), std::ref(trackedYCrCb));

            //Start thread 2 that will handle the HSV color
            thread HSVThread(trackHSVObjects, std::ref(frameColor), threasholdHSV, std::ref(trackedHSV));

            //Start thread 3 that will handle the circle detection
            thread CircleThread(trackCircles, std::ref(frameColor), std::ref(trackedCircles));
            */
            trackObjects(frameColor, threasholdYCrCb, trackedYCrCb, CV_RGB2YCrCb, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX));
            trackObjects(frameColor, threasholdHSV, trackedHSV, CV_RGB2HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX));
            trackCircles(frameColor, frameGray, trackedCircles, true);

            /*
            threasholdYCrCb.copyTo(frameGrayThresholdYCrCb);
            threasholdHSV.copyTo(frameGrayThresholdHSV);
            trackCircles(threasholdYCrCb, frameGrayThresholdYCrCb, trackedCirclesYCrCb, false);
            trackCircles(threasholdHSV, frameGrayThresholdHSV, trackedCirclesHSV, false);
            */

            try
            {
                //YCrCbThread.join();
                //HSVThread.join();
                //CircleThread.join();

                //Match the filtered circles from YCrCb and HSV with eachother and set the prio
                //matchObjects(trackedYCrCb, trackedHSV, bothTemp, false);
                //matchObjects(bothTemp, trackedCircles, both, true);


                //matchObjects(trackedYCrCb, trackedCirclesYCrCb, tempVector, false);
                //matchObjects(trackedHSV, trackedCirclesHSV, tempVector, false);

                matchObjects(trackedYCrCb, trackedHSV, bothTemp, false);
                matchObjects(bothTemp, trackedCircles, both, true);

                //If there is three objects from the previous loop
                int i = 0;
                list< vector<Object> >::iterator iterator;
                for (iterator = lastPrios.begin(); iterator != lastPrios.end(); iterator++){
              //  for (list<vector<Object>>::iterator iter = lastPrios.begin() ; iter != lastPrios.end(); iter++){
                    matchLast(*iterator, both, i);
                    i++;
                }

                //Set the distance for X, Y and Z for all the objects in the both vector
                calculate3DPosition(both, frameColor, ballRadius, FOV_H, FOV_V);

                //Create pairs by their distance
                createPairsByDistance(both, neighborhood, frameColor);

                //Match the pairs into triangles
                matchTriangles(both, neighborhood);

                //Sort the "both" vector with the highest prio first
                std::sort(both.begin(), both.end(), sorting);

                //Move the three highest prioritized circles into the buffer, making use of them in the next loop
                if(lastPrios.size() >= 5)
                {
                    lastPrios.pop_back();
                }

                if(both.size() >= 3)
                {
                    vector<Object> lastPrio;
                    copyObject(both.at(0), lastPrio);
                    copyObject(both.at(1), lastPrio);
                    copyObject(both.at(2), lastPrio);
                    lastPrios.push_front(lastPrio);
                }
/*
                if(both.size() >= 3 && dflag >= 3)
                {
                    Object temp = both.at(2);
                    Object temp2 = both.at(2);
                    Object temp3 = both.at(2);
                    for(int i = 0; i < 3; ++i)
                    {
                        if(both.at(i).getXPos() <= temp.getXPos())
                        {
                            temp = both.at(i);
                        }
                        else if(both.at(i).getXPos() <= temp2.getXPos())
                        {
                            temp2 = both.at(i);
                        }
                        else
                        {
                            temp3 = both.at(i);
                        }

                    }
                    both.at(0) = temp;
                    both.at(1) = temp2;
                    both.at(2) = temp3;
                }

*/
                calculatePlane(both, midPos, angles, frameColor, ballRadius, FOV_H, dflag);

                //HERE WE HAVE OUR AWESOME VALUES. THEY LAY IN midPos AND angles!! OMG ERMAHGERD!
                sure = both.at(0).getPrio() + both.at(1).getPrio() + both.at(2).getPrio();
                sure /= 360;


                if(rflag == 1)
                {
                    cout << midPos[2] << endl;
                }
                else if(rflag == 2)
                {
                    cout << midPos[0] << DELIMITER_RECORD << midPos[1] << DELIMITER_RECORD << midPos[2] << endl;
                }
                else if(rflag == 3)
                {

                    cout << midPos[0] << DELIMITER_RECORD << midPos[1] << DELIMITER_RECORD << midPos[2] << DELIMITER_GROUP << angles[0] << DELIMITER_RECORD << angles[1] << DELIMITER_RECORD << angles[2] << DELIMITER_GROUP << sure << endl;
                }

                if(dflag >= 1)
                {
                    if(angleBuff.size() > 50)
                    {
                        angleBuff.clear();
                        distanceBuff.clear();
                    }

                    if(midPos.size() == 3 && angles.size() == 3)
                    {
                        angleBuff.push_back(angles.at(1));
                        distanceBuff.push_back(midPos.at(2));

                        putText(frameColor, "Distance X: " + std::to_string((int)midPos.at(0)) + " cm", cv::Point(30, 30), 1, 1.6, cv::Scalar(0, 255, 255), 2);
                        putText(frameColor, "Distance Y: " + std::to_string((int)midPos.at(1)) + " cm", cv::Point(30, 55), 1, 1.6, cv::Scalar(0, 255, 255), 2);
                        putText(frameColor, "Distance Z: " + std::to_string((int)midPos.at(2)) + " cm", cv::Point(30, 80), 1, 1.6, cv::Scalar(0, 255, 255), 2);

                        putText(frameColor, "Pitch: " + std::to_string((int)angles.at(1)), cv::Point(30, 110), 1, 1.4, cv::Scalar(0, 255, 255), 2);
                        putText(frameColor, "Roll:  " + std::to_string((int)angles.at(2)), cv::Point(30, 130), 1, 1.4, cv::Scalar(0, 255, 255), 2);
                        putText(frameColor, "Yaw:   " + std::to_string((int)angles.at(0)), cv::Point(30, 150), 1, 1.4, cv::Scalar(0, 255, 255), 2);
                    }

                    //FPS
                    putText(frameColor,"FPS: "+std::to_string(fps),Point(30,180), 1, 1.2, cv::Scalar(0, 255, 255), 1);

                    if(!vflag.empty())
                    {
                        //Text if we are recording video or not
                        putText(frameColor,"Recording: "+ std::to_string(writeVideo),Point(30,210), 1, 1.2, cv::Scalar(0, 255, 255), 1);
                    }

                    putText(frameColor, "Sure?:   " + std::to_string((int)(sure*100)), cv::Point(30, 240), 1, 1.4, cv::Scalar(0, 255, 255), 2);

                }

                //If in debug mode, print the prio to the screen
                if(dflag >= 2)
                {
                    
                    int j = 0;
                    list< vector<Object> >::iterator iterator;
                    for (iterator = lastPrios.begin(); iterator != lastPrios.end(); iterator++){
                   // for (list<vector<Object>>::iterator iter = lastPrios.begin() ; iter != lastPrios.end(); iter++){
                        for(int i = 0; i<3; ++i)
                        {
                            circle(frameColor, Point(iterator->at(i).getXPos(), iterator->at(i).getYPos()), iterator->at(i).getRadius(), Scalar(150, 0, 0, 42*(j+1)), 1/(j+1));
                        }
                        j++;
                    }
                    
                    for(int i = 0; i<trackedCircles.size(); ++i)
                    {
                        circle(frameColor, Point(trackedCircles.at(i).getXPos(), trackedCircles.at(i).getYPos()), trackedCircles.at(i).getRadius(), Scalar(255,255,0), 1);
                    }
                    for(int i = 0; i<trackedHSV.size(); ++i)
                    {
                        circle(frameColor, Point(trackedHSV.at(i).getXPos(), trackedHSV.at(i).getYPos()), trackedHSV.at(i).getRadius(), Scalar(255,0,255));
                    }
                    for(int i = 0; i<trackedYCrCb.size(); ++i)
                    {
                        circle(frameColor, Point(trackedYCrCb.at(i).getXPos(), trackedYCrCb.at(i).getYPos()), trackedYCrCb.at(i).getRadius(), Scalar(0,255,255));
                    }
                    printPrio(both, frameColor);
                }




                // Display image
                if(dflag >= 1)
                {
                    imshow(windowTitle, frameColor);
                }
                if(dflag >=2){
                    if(!threasholdYCrCb.empty())
                    {
                        imshow(windowYCrCb, threasholdYCrCb);
                    }
                    if(!threasholdHSV.empty())
                    {
                        imshow(windowHSV, threasholdHSV);
                    }
                    if(!frameGray.empty())
                    {
                        imshow(windowGray, frameGray);
                    }
                    if(!frameColorDark.empty())
                    {
                        imshow("Dark", frameColorDark);
                    }
                    /*if(!frameGrayThresholdHSV.empty())
                    {
                        imshow(windowHSV2, frameGrayThresholdHSV);
                    }
                    if(!frameGrayThresholdYCrCb.empty())
                    {
                        imshow(windowYCrCb2, frameGrayThresholdYCrCb);
                    }*/
                }

                if(!vflag.empty() && writeVideo)
                {
                    outputColor.write(frameColor);
                    outputHSV.write(threasholdHSV);
                    outputYCrCb.write(threasholdYCrCb);
                    outputGray.write(frameGray);
                }
            }
            catch (cv::Exception & e)
            {
                cout << e.what() << endl;
                //return 1;
            }
        }
        char k = waitKey(1);
        if (k == 'q' || k == 'Q')
        {
            loop = false;
            outputColor.release();
            outputHSV.release();
            outputYCrCb.release();
            outputGray.release();
        }
        if (k == 'c' || k == 'C')
        {
            lastPrios.clear();
        }
        if (dflag>=1 && (k=='f' || k=='F')) printFPS=!printFPS;
        if (dflag>=1 && !vflag.empty() && (k=='v' || k=='V')) writeVideo=!writeVideo;
    }

    return 0;
}




