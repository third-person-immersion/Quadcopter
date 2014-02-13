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

	bool getHasCircle();
	void setHasCircle(bool value);


private:

	double xPos, yPos, area, radius, prio;
	string type;
	bool hasCircle;

};

