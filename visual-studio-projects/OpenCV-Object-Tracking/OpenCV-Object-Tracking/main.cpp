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
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/flann.hpp"
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
const std::string imageDir = "images/";         // Directory for images
const std::string videoDir = "videos/";         // Directory for videos
const std::string sourceVideo = "source2.mp4";   // Source video
const std::string objectProfile = "pen.png";    // Object to be detected within video

// Variables
const int minimumHessian = 400;                 // Parameter for feature detector

/*
 * 	Defines
 */
#define EXIT_SUCCESS 0
#define EXIT_ERROR 1
#define MIN_COLOR_VALUE 0
#define MAX_COLOR_VALUE 255

/*
 * 	Function Prototypes
 */
void printHeader();
bool loadImage(Mat&, std::string);
void displayImage(Mat&, std::string, bool);
void displayVideo(VideoCapture&, const std::string);
int detectObjectInVideo(VideoCapture&, Mat*, const std::string);

int main(int argc, const char* argv[])
{

	// Show greeting header
	printHeader();

	// Load images into buffer from video source
	// Start capture using a source video location
	std::string filename = videoDir + sourceVideo;
	VideoCapture capture(filename);

	// Check that it opened properly
	if (!capture.isOpened())
	{
        // Exit with error
		std::cout << "$ ERROR: Unable to open stream from <" << filename << ">." << std::endl;
		std::cin.get();
        return EXIT_ERROR;
    }
    else
    {
        // Provide feedback to let the user know loading was successful
        std::cout << "$ Video source opened from <" << filename << ">." << std::endl;
    }

    // Load object profile that will be detected
    Mat objectToDetect;
    if(!loadImage(objectToDetect, objectProfile)) 
    {
        std::cin.get();
        return EXIT_ERROR;
    }

    // Perform 2D homography on the video frames and draw bounding boxes on predictions
    // of where the object is
    const std::string detectionWindow = "Homography-Results";
    namedWindow(detectionWindow, 1);
    detectObjectInVideo(capture, &objectToDetect, detectionWindow);

	// Exit successfully
	std::cout << "$ Program terminated successfully." << std::endl;
    std::cin.get();                                                 // Wait for keypress
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
	input = imread(imageDir + filename, 0); // Grayscale

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
void displayVideo(VideoCapture& capture, const std::string windowName)
{

	// Loop through video frames
	Mat frame, frameGray;
	int count = 0;
	while(1)
	{
		capture >> frame;			                // Get current frame image
		if (frame.empty()) break;	                // Check for end of video
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
		imshow(windowName, frameGray);	            // Display frame in window
		waitKey(20);                                // 20ms delay between frame displays
		count++;
	}
	waitKey(0);					                    // Press key to close window

}

/*
 *
 * 	detectObject - uses 2D homography to find a given object within a video; draws a bounding box around
 *  the object and displays the frames of the video to the screen; returns number of frames in the video
 *
 */
int detectObjectInVideo(VideoCapture& capture, Mat* objectToDetect, const std::string windowName)
{

    
    // Loop variables
    Mat frame;
    int count = 0;

    // Feature detection variables
    Mat frameGray, objectDescriptors, frameDescriptors, combinedMatches, homography;
    SurfFeatureDetector detector(minimumHessian);
    std::vector<KeyPoint> objectKeypoints, frameKeypoints;
    SurfDescriptorExtractor extractor;
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches, keptMatches;
    double maxDistance = 0.0;
    double minDistance = 100.0;
    double thisDistance;
    std::vector<Point2f> objectPoints, framePoints;
    std::vector<Point2f> objectCorners(4);
    std::vector<Point2f> frameCorners(4);

    // Pre-compute object keypoints and descriptors
    detector.detect(*objectToDetect, objectKeypoints);
    extractor.compute(*objectToDetect, objectKeypoints, objectDescriptors);

    // For each video frame, convert it to grayscale and process it
    while (1)
    {
        capture >> frame;			                // Get current frame image
		if (frame.empty()) break;	                // Check for end of video
        cvtColor(frame, frameGray, COLOR_BGR2GRAY); // Convert frame to grayscale

        // Cleanup
        objectPoints.clear();
        framePoints.clear();
        matches.clear();
        keptMatches.clear();

        // Detect keypoints and calculate feature vectors of video frame
        detector.detect(frameGray, frameKeypoints);
        extractor.compute(frameGray, frameKeypoints, frameDescriptors);

        // FLANN matching between descriptors
        matcher.match(objectDescriptors, frameDescriptors, matches);

        // Calculate minimum and maximum distance between keypoints
        thisDistance = 0.0;
        for (int i = 0; i < objectDescriptors.rows; i++)
        {
            // Update min and max distances as needed
            thisDistance = matches[i].distance;
            if (thisDistance < minDistance) minDistance = thisDistance;
            if (thisDistance > maxDistance) maxDistance = thisDistance;
        }

        // Keep only descriptor matches that are within 3 times the minimum distance
        for (int i = 0; i < objectDescriptors.rows; i++)
        {
            if (matches[i].distance < 1.1 * max(objectToDetect->rows, objectToDetect->cols)) keptMatches.push_back(matches[i]);
        }

        // Draw the remaining matches
        drawMatches(*objectToDetect, objectKeypoints, frameGray, frameKeypoints, keptMatches, combinedMatches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Localize the object
        for (int i = 0; i < keptMatches.size(); i++)
        {
            objectPoints.push_back(objectKeypoints[keptMatches[i].queryIdx].pt);
            framePoints.push_back(frameKeypoints[keptMatches[i].trainIdx].pt);
        }

        // Find homography
        homography.release();
        homography = findHomography(Mat(objectPoints), Mat(framePoints), RANSAC, 0.1);
        if (homography.empty()) continue;

        // Corners of source image for object profile
        objectCorners[0] = Point2f(0, 0);
        objectCorners[1] = Point2f(objectToDetect->cols, 0);
        objectCorners[2] = Point2f(objectToDetect->cols, objectToDetect->rows);
        objectCorners[3] = Point2f(0, objectToDetect->rows);

        // Perspective drawing
        perspectiveTransform(objectCorners, frameCorners, homography);

        // Draw lines between corners 1 -> 2, 2 -> 3... etc
        line(combinedMatches, frameCorners[0] + Point2f(objectToDetect->cols, 0), frameCorners[1] + Point2f(objectToDetect->cols, 0), Scalar(0, 255, 0), 4);
        line(combinedMatches, frameCorners[1] + Point2f(objectToDetect->cols, 0), frameCorners[2] + Point2f(objectToDetect->cols, 0), Scalar(0, 255, 0), 4);
        line(combinedMatches, frameCorners[2] + Point2f(objectToDetect->cols, 0), frameCorners[3] + Point2f(objectToDetect->cols, 0), Scalar(0, 255, 0), 4);
        line(combinedMatches, frameCorners[3] + Point2f(objectToDetect->cols, 0), frameCorners[0] + Point2f(objectToDetect->cols, 0), Scalar(0, 255, 0), 4);

        // Show results
		imshow(windowName, combinedMatches);	    // Display frame in window
        waitKey(10);
		count++;

	}
	waitKey(0);					                    // Press key to close window

    // Return number of frames in video
    return count;

}
