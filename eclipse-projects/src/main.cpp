/*
 *
 * 	File: 			main.cpp
 * 	Author: 		Dillon Fisher
 * 	Date: 			7 April, 2014
 * 	Description:	Driver file for object detection
 * 	Usage:
 *
 */

/*
 * 	Includes
 */
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

/*
 * 	Globals
 */
const std::string imageDir = "images/";

/*
 * 	Defines
 */
#define EXIT_SUCCESS 0
#define EXIT_ERROR 1

/*
 * 	Namespace
 */
using namespace cv;

/*
 * 	Function Prototypes
 */
void printHeader();

int main(int argc, char* const argv[])
{

	// Show greeting header
	printHeader();

	std::string blah;
	std::cin >> blah;

	// Exit successfully
	return EXIT_SUCCESS;

}

/*
 *
 * 	printHeader - prints a short greeting header identifying the program
 *
 */
void printHeader()
{

	std::cout << "===== OpenCV Object Recognition =====" << std::endl;

}
