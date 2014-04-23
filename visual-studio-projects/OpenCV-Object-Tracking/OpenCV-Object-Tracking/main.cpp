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
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>

/*
 * 	Namespace
 */
using namespace cv;

/*
 * 	Globals
 */
// Filenames
const std::string imageDir = "images/";
const std::string videoDir = "videos/";
const std::string sourceVideo = "source.mp4";

/*
 * 	Defines
 */
#define EXIT_SUCCESS 0
#define EXIT_ERROR 1

/*
 * 	Function Prototypes
 */
void printHeader();
bool loadImage(Mat&, std::string);
void displayImage(Mat&, std::string, bool);
void displayVideo(VideoCapture&, std::string);
void threshholdImage(Mat&, Mat&);
int displayThreshholdedFrames(VideoCapture&, const std::string&);

int main(int argc, char* const argv[])
{

	// Show greeting header
	printHeader();

	// Load images into buffer from video source
	// Start capture using a source video location
	std::string filename = videoDir + sourceVideo;
	VideoCapture capture("source.mp4");

	// Check that it opened properly
	if (!capture.isOpened())
	{
        // Exit with error
		std::cout << "$ Unable to open stream from <" << filename << ">." << std::endl;
		std::cin.get();
        return EXIT_ERROR;
    }

	// Loop through the video frames, threshhold them and display them
	namedWindow("Threshholded-Frames", 1);
	int totalFrames = displayThreshholdedFrames(capture, "Threshholded-Frames");
    std::cout << "$ Total frames in video source: " << totalFrames << std::endl;

	// Exit successfully
	std::cout << "$ Program terminated successfully." << std::endl;
    std::cin.get();
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

/*
 *
 * 	loadImage - loads an image from file into the specified matrix
 *
 */
bool loadImage(Mat &input, std::string filename)
{

	// Verify that a filename was actually entered
	if (filename.length() < 1)
	{
		std::cout << "$ ERROR: Invalid filename and path entered." << std::endl;
		return EXIT_ERROR;
	}

	// Load desired image
	input = imread(imageDir + filename, 1);

	// Check that it loaded correctly
	if (!input.data)
	{
		std::cout << "$ ERROR: Failed to load requested image. Exiting..." << std::endl;
		std::cout << "$ NOTE: Image '" << filename << "' must be in '<project_dir>/images'." << std::endl;
		return false;
	}
	else
	{
		// Everything went smoothly
		std::cout << "$ Image <" << filename << "> was successfully loaded." << std:: endl;
		return true;
	}

}

/*
 *
 * 	displayImage - displays the desired image in a window and waits for a key to be
 * 	pressed if desired
 *
 */
void displayImage(Mat& input, std::string windowName, bool waitForKey)
{

	// Show image in created window
	namedWindow(windowName, 1);
	imshow(windowName, input);
	if (waitForKey) waitKey(0);

}

/*
 *
 * 	displayVideo - shows the desired video file in a window and waits for the user to
 * 	press a key before closing it
 *
 */
void displayVideo(VideoCapture& capture, std::string windowName)
{

	// Loop through video frames
	Mat frame;
	int count = 0;
	while(1)
	{
		capture >> frame;			// Get current frame image
		if (count == 0)
		{
			Mat* output = new Mat(frame.size(), CV_8U);
			threshholdImage(frame, *output);
			displayImage(*output, "testTreshhold", true);
		}
		if (frame.empty()) break;	// End of video
		imshow(windowName, frame);	// Display frame in window
		waitKey(20);
		count++;
	}
	waitKey(0);					// Press key to close window

}

/*
 *
 * 	threshholdImage - returns a binary threshholded version of the input image
 *
 */
void threshholdImage(Mat& input, Mat& output)
{
	// Perform the threshholding operation by checking if each pixel is in the proper
	// range of values
	inRange(input, Scalar(20, 20, 0), Scalar(80, 80, 15), output);

}

/*
 *
 * 	displayThreshholdedFrames - uses a window to display, frame-by-frame, the contents of a
 *  video given a certain threshhold
 *
 */
int displayThreshholdedFrames(VideoCapture& capture, const std::string& windowName)
{
	
    // Grab each frame in the video, threshhold it and display it
    Mat frame, threshholdedFrame;
	int frameCount = 0;
	while(1)
	{
        // Get current frame image
		capture >> frame;

        // Check for end conditions
		if (frame.empty()) return frameCount;	// End of video
        else
        {
            // Apply threshholding routine
		    threshholdImage(frame, threshholdedFrame);
		    displayImage(threshholdedFrame, windowName, false);
		    waitKey(20);            // Wait 20ms to display next frame
		    frameCount++;           // Update frame counter
        }
	}

}
