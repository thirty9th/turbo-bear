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

/* yl - will clean up later, just stuck here to get working ver.
  set as global var to avoid params 
*/
//
  Mat     result, result_cropped, prev_frame, current_frame, next_frame;
  Mat     d1, d2, motion;
  int number_of_changes, number_of_sequence = 0;
  Scalar mean_, color(0,255,255); // yellow
  int x_start, x_stop, y_start , y_stop;
  int there_is_motion = 5;
  int max_deviation = 20;
  Mat kernel_ero = getStructuringElement(MORPH_RECT, Size(2,2));
  bool    vcapBool, keepGoing;
//

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
void detectHueColor ( const Mat &, Mat &);    //detect main hue color
void detectBGRColor ( const Mat &, Mat &);    //detect main BGR color
void initSome(VideoCapture&);
int detectMotion(const Mat & , const Mat &, Mat & , Mat & , int , int , int , int , int, Scalar &);
void detectBGRColor (const Mat & );
void motionCheck(const Mat& , Mat&);


int main(int argc, const char* argv[])
{

	// Show greeting header
	printHeader();

	// Load images into buffer from video source
	// Start capture using a source video location
	std::string filename = videoDir + sourceVideo;
	VideoCapture capture(filename);
  Mat frame;

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

    std::cout << "fps: " << capture.get(CV_CAP_PROP_FPS)<< "\t w: " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << "\t h: "<< capture.get(CV_CAP_PROP_FRAME_HEIGHT) << "\n";
    initSome(capture);
    while ( capture.isOpened() ) {//main Loop

      capture.read(frame);//read in next frame
      result=frame;
      putText(result,"HUE",Point(0,result.rows-75),FONT_HERSHEY_DUPLEX,1,Scalar(255,255,255));
      putText(result,"RGB",Point(100,result.rows-75),FONT_HERSHEY_DUPLEX,1,Scalar(255,255,255));
      motionCheck(frame, result);//motion detection proc and rgb and hue of detected area

    /*yl-commented out for now till coded to work within main loop
    where frame is clean iamge to process, result is image written to show results, currently has rect. from motion
    recommend: detectObjectInVideo( &objectToDetect, detectionWindow, frame, result);
    detectObjectInVideo(capture, &objectToDetect, detectionWindow);
    */
      imshow(detectionWindow,result);
      if (waitKey(20) == 27) { std::cout << "esc key is pressed by user" << "\n"; break; };

    };//while main Loop

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

// initialize some settings before main loop
void initSome(VideoCapture& capture) {
  //capture.read(result);
  capture.read(prev_frame); cvtColor(prev_frame, prev_frame, CV_BGR2GRAY);
  capture.read(current_frame); cvtColor(current_frame, current_frame, CV_BGR2GRAY);
  capture.read(next_frame); cvtColor(next_frame, next_frame, CV_BGR2GRAY);
  x_start=y_start=10;
  x_stop=current_frame.cols-10;
  y_stop=current_frame.rows-10;
};//initSome


int detectMotion(const Mat & motion, const Mat & frame, Mat & result, Mat & result_cropped,
                 int x_start, int x_stop, int y_start, int y_stop,
                 int max_deviation,
                 Scalar & color)
{
    // calculate the standard deviation
    Scalar mean, stddev;
    meanStdDev(motion, mean, stddev);
    // if not to much changes then the motion is real (neglect agressive snow, temporary sunlight)
    if(stddev[0] < max_deviation) {
        int number_of_changes = 0;
        int min_x = motion.cols, max_x = 0;
        int min_y = motion.rows, max_y = 0;
        // loop over image and detect changes
        for(int j = y_start; j < y_stop; j+=2){ // height
            for(int i = x_start; i < x_stop; i+=2){ // width
                // check if at pixel (j,i) intensity is equal to 255
                // this means that the pixel is different in the sequence
                // of images (prev_frame, current_frame, next_frame)
                if(static_cast<int>(motion.at<uchar>(j,i)) == 255)
                {
                    number_of_changes++;
                    if(min_x>i) min_x = i;
                    if(max_x<i) max_x = i;
                    if(min_y>j) min_y = j;
                    if(max_y<j) max_y = j;
                }
            }
        }
        if(number_of_changes){
            //check if not out of bounds
            if(min_x-10 > 0) min_x -= 10;
            if(min_y-10 > 0) min_y -= 10;
            if(max_x+10 < frame.cols-1) max_x += 10;
            if(max_y+10 < frame.rows-1) max_y += 10;
            // draw rectangle round the changed pixel
            Point x(min_x,min_y);
            Point y(max_x,max_y);
            Rect rect(x,y);
            Mat cropped = frame(rect);
            cropped.copyTo(result_cropped);
            rectangle(result,rect,color,1);
        };//
        return number_of_changes;
    };//
    return 0;
};//detectMotion


//detect main Hue color
//based code off of
// http://laconsigna.wordpress.com/2011/04/29/1d-histogram-on-opencv/
void detectHueColor ( const Mat & tmat, Mat & result ) {
//find main hue color
  Mat hsv;
  vector <Mat> channels;
  int i, h;
  MatND tH;
  int maxH;
  int nbins=64;
  int histSize[]= {nbins};
  float hue_range[] = { 0, 180 };
  const float* hRanges[] = { hue_range };

  cvtColor(tmat, hsv, CV_BGR2HSV);
  split(hsv, channels);
  calcHist(&channels[0], 1, 0, Mat(), tH, 1, histSize, hRanges, true, false); 
  h=0; maxH=0;
  
  for ( i=0; i<tH.rows-1; i++ ) {
    if (tH.at<float>(i) > maxH ) {
      maxH=(int) tH.at<float>(i) ;
      h=i*180/tH.rows;
    };//if greater
  };//for i
  
//  cout << "hue: " << h << "  maxh : " << maxH << endl;

//display hue color 50x50 max value, saturation
  Mat hmColor(50,50,CV_8UC3,Scalar(h,255,255) );
  Mat mHColor;
  cvtColor(hmColor, mHColor, CV_HSV2BGR);
//  imshow("Main Hue Color", mHColor);

 mHColor.copyTo(result.colRange(0,50).rowRange(result.rows-50,result.rows) );

};//detect Color


void detectBGRColor ( const Mat & tmat, Mat & result) {
//based code off of
//http://stackoverflow.com/questions/20567643/getting-dominant-colour-value-from-hsv-histogram
  Mat image_bgr;
  cvtColor(tmat, image_bgr, CV_BGR2HSV);
    
  int bbins = 36, gbins = 36, rbins = 36;
  int histSize[] = {bbins, gbins, rbins};
  float branges[] = { 0, 256 }; float granges[] = { 0, 256 }; float rranges[] = { 0, 256 };
  const float* ranges[] = { branges, granges, rranges };
  MatND hist;
  int channels[] = {0, 1, 2};

  calcHist( &image_bgr, 1, channels, Mat(), // do not use mask
           hist, 3, histSize, ranges,
           true, // the histogram is uniform
           false );
  
  int maxVal=0, bMax=0, gMax=0, rMax=0;

  for( int b = 0; b < bbins; b++ ) {
    for( int g = 0; g < gbins; g++ ) {
      for( int r = 0; r < rbins; r++ ) {
        int binVal = hist.at<int>(b, g, r);
        if(binVal > maxVal) {
          maxVal = binVal;
          bMax = b*256/bbins; gMax = g*256/gbins; rMax = r*256/rbins;
        };//if
      };//for r
    };//for g
  };//for b

//show result
//cout<< "b: " << bMax << "\tg: " << gMax << " \tr: " << rMax << endl;
  Mat mBGRColor(50,50, CV_8UC3, Scalar(bMax, gMax, rMax) );
//  imshow("Main BGR Color", mBGRColor);
  mBGRColor.copyTo(result.colRange(100,150).rowRange(result.rows-50,result.rows) );

};//detect color


void motionCheck(const Mat & frame, Mat & result ) {
  prev_frame = current_frame; current_frame = next_frame; next_frame=frame;
  cvtColor(next_frame, next_frame, CV_BGR2GRAY);
    
  absdiff(prev_frame, next_frame, d1);
  absdiff(next_frame, current_frame, d2);
  bitwise_and(d1, d2, motion);
  threshold(motion, motion, 35, 255, CV_THRESH_BINARY);
  erode(motion, motion, kernel_ero);
    
  number_of_changes = detectMotion(motion, frame, result, result_cropped,  x_start, x_stop, y_start, y_stop, max_deviation, color);
    if(number_of_changes>=there_is_motion) {
      if(number_of_sequence>0) { 
        //cout << "changes detected" << endl;
//              imshow("Main",result);
//              imshow("Cropped",result_cropped);
        detectHueColor(result_cropped,result); //detect main hue color
        detectBGRColor(result_cropped,result); // detect main BGR color
      };//if 
        number_of_sequence++;
    } else {
      number_of_sequence = 0;
    };//else
};//motionCheck