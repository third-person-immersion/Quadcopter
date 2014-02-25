#include "Object.h"

Object::Object(void)
{
    Object::prio = 0.0;
    Object::matches = 0;
    Object::matchesPair = 0;
    Object::added = false;
    Object::checked = false;
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

int Object::getMatches(){
    return Object::matches;
}

void Object::incMatches(){
    Object::matches++;
}

int Object::getMatchesPair(){
    return Object::matchesPair;
}

void Object::incMatchesPair(){
    Object::matchesPair++;
}


bool Object::getAdded(){
    return Object::added;
}
   
void Object::setAdded(bool value){
    Object::added = value;
}

double Object::getXDist(){
    return Object::xDist;
}
void Object::setXDist(double dist){
    Object::xDist = dist;
}
    
double Object::getYDist(){
    return Object::yDist;
}
void Object::setYDist(double dist){
    Object::yDist = dist;
}
    
double Object::getZDist(){
    return Object::zDist;
}
void Object::setZDist(double dist){
    Object::zDist = dist;
}

bool Object::getChecked(){
    return Object::checked;
}
   
void Object::setChecked(bool value){
    Object::checked = value;
}