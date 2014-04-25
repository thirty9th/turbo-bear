//ver 1.4

//http://blog.cedric.ws/opencv-simple-motion-detection
//requires webcam

#define _CRT_SECURE_NO_WARNINGS

//
//  Created by Cedric Verstraeten on 18/02/14.
//  Copyright (c) 2014 Cedric Verstraeten. All rights reserved.
//

#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void detectHueColor ( const Mat & );    //detect main hue color
void detectBGRColor ( const Mat & );    //detect main BGR color

// Check if there is motion in the result matrix
// count the number of changes and return.
inline int detectMotion(const Mat & motion, Mat & result, Mat & result_cropped,
                 int x_start, int x_stop, int y_start, int y_stop,
                 int max_deviation,
                 Scalar & color)
{
    // calculate the standard deviation
    Scalar mean, stddev;
    meanStdDev(motion, mean, stddev);
    // if not to much changes then the motion is real (neglect agressive snow, temporary sunlight)
    if(stddev[0] < max_deviation)
    {
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
            if(max_x+10 < result.cols-1) max_x += 10;
            if(max_y+10 < result.rows-1) max_y += 10;
            // draw rectangle round the changed pixel
            Point x(min_x,min_y);
            Point y(max_x,max_y);
            Rect rect(x,y);
            Mat cropped = result(rect);
            cropped.copyTo(result_cropped);
            rectangle(result,rect,color,1);
        }
        return number_of_changes;
    }
    return 0;
}



int main (int argc, char * const argv[])
{
    const int DELAY = 500; // in mseconds, take a picture every 1/2 second

    bool fileSuccess;

/*
    // Set up camera
    CvCapture * camera = cvCaptureFromCAM(CV_CAP_ANY);
    cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH, 1280); // width of viewport of camera
    cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT, 720); // height of ...
*/

    // Take images and convert them to gray
    Mat result, result_cropped;

    /*
    Mat prev_frame = result = cvQueryFrame(camera);
    Mat current_frame = cvQueryFrame(camera);
    Mat next_frame = cvQueryFrame(camera);
    */

//http://opencv-srf.blogspot.com/2011/09/capturing-images-videos.html
    VideoCapture cap("../../MVI_0182.MOV");
    if ( !cap.isOpened() ) {
      cout << "Cannot open the video file" << endl;
      return -1;
    };
    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    cout << "Frame per seconds : " << fps << endl;
    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

    Mat prev_frame, current_frame, next_frame;
    cap.read(prev_frame);
    cap.read(current_frame);
    cap.read(next_frame);

    cvtColor(current_frame, current_frame, CV_RGB2GRAY);
    cvtColor(prev_frame, prev_frame, CV_RGB2GRAY);
    cvtColor(next_frame, next_frame, CV_RGB2GRAY);
    
    // d1 and d2 for calculating the differences
    // result, the result of and operation, calculated on d1 and d2
    // number_of_changes, the amount of changes in the result matrix.
    // color, the color for drawing the rectangle when something has changed.
    Mat d1, d2, motion;
    int number_of_changes, number_of_sequence = 0;
    Scalar mean_, color(0,255,255); // yellow
    
    // Detect motion in window
    int x_start = 10, x_stop = current_frame.cols-11;
    int y_start = 350, y_stop = 530;

    // If more than 'there_is_motion' pixels are changed, we say there is motion
    // and store an image on disk
    int there_is_motion = 5;
    
    // Maximum deviation of the image, the higher the value, the more motion is allowed
    int max_deviation = 20;
    
    // Erode kernel
    Mat kernel_ero = getStructuringElement(MORPH_RECT, Size(2,2));

    namedWindow("moved",0);
    namedWindow("cropped");
    namedWindow("Main Hue Color");
    namedWindow("Main BGR Color");
    resizeWindow("moved",640,480);
//    resizeWindow("cropped",200,200);
    moveWindow("moved",10,10);
    moveWindow("cropped",500,500);
    moveWindow("Main Hue Color",500,50);
    moveWindow("Main BGR COlor", 700, 150);
    
    // All settings have been set, now go in endless loop and
    // take as many pictures you want..
    while (true){
        // Take a new image
        prev_frame = current_frame;
        current_frame = next_frame;
  //      next_frame = cvQueryFrame(camera);
        fileSuccess=cap.read(next_frame);
        if (!fileSuccess) {
          cout<<"something went wrong with file"<<endl;
        };

        result = next_frame;
        cvtColor(next_frame, next_frame, CV_RGB2GRAY);

        // Calc differences between the images and do AND-operation
        // threshold image, low differences are ignored (ex. contrast change due to sunlight)
        absdiff(prev_frame, next_frame, d1);
        absdiff(next_frame, current_frame, d2);
        bitwise_and(d1, d2, motion);
        threshold(motion, motion, 35, 255, CV_THRESH_BINARY);
        erode(motion, motion, kernel_ero);
        
        number_of_changes = detectMotion(motion, result, result_cropped,  x_start, x_stop, y_start, y_stop, max_deviation, color);
        
        // If a lot of changes happened, we assume something changed.
        if(number_of_changes>=there_is_motion)
        {
            if(number_of_sequence>0){ 
              /*
                saveImg(result,DIR,EXT,DIR_FORMAT.c_str(),FILE_FORMAT.c_str());
                saveImg(result_cropped,DIR,EXT,DIR_FORMAT.c_str(),CROPPED_FILE_FORMAT.c_str());
              */

              cout << "changes detected" << endl;
              imshow("moved",result);
              imshow("cropped",result_cropped);
              detectHueColor(result_cropped); //detect main hue color
              detectBGRColor(result_cropped); // detect main BGR color
            }
            number_of_sequence++;
        }
        else
        {
            number_of_sequence = 0;
            // Delay, wait a 1/2 second.
            cvWaitKey (DELAY);
        }
    }
    return 0;    
}

//detect main Hue color
// http://laconsigna.wordpress.com/2011/04/29/1d-histogram-on-opencv/
void detectHueColor ( const Mat & tmat ) {
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
  
  cout << "hue: " << h << "  maxh : " << maxH << endl;

//display hue color 50x50 max value, saturation
  Mat hmColor(50,50,CV_8UC3,Scalar(h,255,255) );
  Mat mHColor;
  cvtColor(hmColor, mHColor, CV_HSV2BGR);
  imshow("Main Hue Color", mHColor);

};//detect Color


void detectBGRColor ( const Mat & tmat) {
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

  //cout<< "b: " << bMax << "\tg: " << gMax << " \tr: " << rMax << endl;

  Mat mBGRColor(50,50, CV_8UC3, Scalar(bMax, gMax, rMax) );
  imshow("Main BGR Color", mBGRColor);
};//detect color
