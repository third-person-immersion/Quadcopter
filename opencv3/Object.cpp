//#include "stdafx.h"
#include "Object.h"

Object::Object(void)
{
}


Object::~Object(void)
{
}


double Object::getXPos(){

	return Object::xPos;

}

void Object::setXPos(double x){

	Object::xPos = x;

}

double Object::getYPos(){

	return Object::yPos;

}

void Object::setYPos(double y){

	Object::yPos = y;

}

double Object::getArea(){

	return Object::area;

}

void Object::setArea(double a){

	Object::area = a;

}

double Object::getRadius(){

	return Object::radius;

}

void Object::setRadius(double r){

	Object::radius = r;

}
string Object::getType(){

	return Object::type;

}

void Object::setType(string t){

	Object::type = t;

}

double Object::getPrio(){
	return Object::prio;
}

void Object::incPrio(double prio){
	Object::prio+=prio;
}

void Object::decPrio(double prio){
	Object::prio-=prio;
}

void Object::setPrio(double prio){
	Object::prio = prio;
}

bool Object::getHasCircle(){
	return Object::hasCircle;
}

void Object::setHasCircle(bool value){
	Object::hasCircle = value;
}
