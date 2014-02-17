#pragma once
#include <string>
using namespace std;

class Object
{
public:
	Object(void);
	~Object(void);

	double getXPos();
	void setXPos(double x);

	double getYPos();
	void setYPos(double y);

	double getArea();
	void setArea(double a);

	double getRadius();
	void setRadius(double a);

	string getType();
	void setType(string t);

	double getPrio();
	void incPrio(double prio);
	void decPrio(double prio);
	void setPrio(double prio);

	bool getMatches();
	void incMatches();

    bool getAdded();
    void setAdded(bool value);

    double getXDist();
    void setXDist(double dist);
    
    double getYDist();
    void setYDist(double dist);
    
    double getZDist();
    void setZDist(double dist);

private:

	double xPos, yPos, area, radius, prio, matches, xDist, yDist, zDist;
	string type;
	bool added;

};

