#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

#define POS_INTERVAL 10
#define NEG_INTERVAL 10

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
	blobs.clear();

	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground

	cv::Mat label_image;
	binary.convertTo(label_image, CV_32SC1);

	int label_count = 2; // starts at 2 because 0,1 are used already

	for(int y=0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		for(int x=0; x < label_image.cols; x++) {
			if(row[x] != 1) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

			std::vector <cv::Point2i> blob;

			for(int i=rect.y; i < (rect.y+rect.height); i++) {
				int *row2 = (int*)label_image.ptr(i);
				for(int j=rect.x; j < (rect.x+rect.width); j++) {
					if(row2[j] != label_count) {
						continue;
					}

					blob.push_back(cv::Point2i(j,i));
				}
			}

			blobs.push_back(blob);

			label_count++;
		}
	}
}


void showImg(char *name, Mat frame)
{
	namedWindow(name);
	imshow(name,frame);
}

Mat getSkin(Mat frame)
{
	//vector<Mat> chanels;
	Mat skin,fg;
	cvtColor(frame,fg,CV_RGB2YCrCb);


//	inRange(fg,Scalar(0,85,140),Scalar(255,120,170),skin);
		inRange(fg,Scalar(35,35,35),Scalar(255,255,255),skin);

	return skin;
}

Mat getSkin2(Mat frame)
{
	vector<Mat> chanels;
	Mat skin,fg;
	cvtColor(frame,fg,CV_RGB2YCrCb);
	split(fg,chanels);
	showImg("Y",chanels[0]);
	showImg("Cr",chanels[1]);
	showImg("Cb",chanels[2]);
for (int i = 0; i < fg.rows; i++)
{
	for (int j = 0; j < fg.cols; j++)
	{
		if ((chanels[1].at<uchar>(i,j) < 177) && (chanels[1].at<uchar>(i,j) > 137) && (chanels[2].at<uchar>(i,j) < 127) && (chanels[2].at<uchar>(i,j) > 77) && (chanels[2].at<uchar>(i,j) + 0.6 * chanels[1].at<uchar>(i,j) > 190) && (chanels[2].at<uchar>(i,j) + 0.6 * chanels[1].at<uchar>(i,j) < 215))
		 chanels[0].at<uchar>(i,j)=255;
		else
		 chanels[0].at<uchar>(i,j)=0;
	}
}
	return chanels[0];
}

void test(Mat* img)
{
 for(int i = 0; i < (*img).rows; i++)
	 for(int j = 0; j < (*img).cols; j++)
	 {
		(*img).at<Vec3b>(i,j)[0] = 100;
			 (*img).at<Vec3b>(i,j)[0] = 200;
			 (*img).at<Vec3b>(i,j)[0] = 150;
	 }
}

BackgroundSubtractorMOG2 mog(100,50,false);

int hand()
{
	
	VideoCapture cap(0);
	Mat frame,fg,skin,img;
	vector<Mat> chanels;
	
	cv::Mat binary;
	std::vector < std::vector<cv::Point2i > > blobs;

	 
	if(!cap.isOpened())
	{
	    printf("Eroare la deschiderea camerei\n");
		return -1;
	}

	Mat element = getStructuringElement( MORPH_RECT,Size( 2*1 + 1, 2*1+1 ),	Point( 1, 1 ) );


	while(cap.read(frame))
	{
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

	
	img = frame.clone();
	 skin = getSkin(frame);
	GaussianBlur(skin,skin,Size(0,0),5);
		 showImg("Skin",skin);
	 // pentru blob
	 // threshold(skin,fg,150,1.0,THRESH_BINARY);

	  findContours( skin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	  
	  vector<vector<Point> >hull( contours.size());
	   vector<vector<int> >hullI( contours.size());
	  vector<vector<Vec4i> >defect(contours.size());

//	  printf("Contour size %d\n",contours.size());

	  for( int i = 0; i < contours.size(); i++ )
	  {  
		  convexHull( Mat(contours[i]), hull[i], false ); 
		   convexHull( Mat(contours[i]), hullI[i], false ); 
		    convexityDefects(contours[i],hullI[i],defect[i]);
			
		//	printf("Defects %d\n",defect[i].size());
			
			for (int j =0; j < defect[i].size(); j++)
			{
				printf("Defect %d and size %d\n",defect[i][j][2],defect[i][j][3]);
				if (defect[i][j][3] > 1500)
				{
			    	line(img,  cv::Point(contours[i][defect[i][j][2]].x,contours[i][defect[i][j][2]].y ), cv::Point(contours[i][defect[i][j][2]].x,contours[i][defect[i][j][2]].y ), cv::Scalar(255, 255, 0),10);
					line(img,  cv::Point(contours[i][defect[i][j][0]].x,contours[i][defect[i][j][0]].y ), cv::Point(contours[i][defect[i][j][0]].x,contours[i][defect[i][j][0]].y ), cv::Scalar(0, 255, 0),10);
				//	line(img,  cv::Point(contours[i][defect[i][j][1]].x,contours[i][defect[i][j][1]].y ), cv::Point(contours[i][defect[i][j][1]].x,contours[i][defect[i][j][1]].y ), cv::Scalar(0, 255, 0),10);
				}
			}

		/*
			for (int j=0; j<hull[i].size();j++)
			{
				printf("HUll #%d %d %d\n",j,contours[i][hullI[i][j]].x,contours[i][hullI[i][j]].y);
				line(img,  cv::Point(contours[i][hullI[i][j]].x,contours[i][hullI[i][j]].y ), cv::Point(contours[i][hullI[i][j]].x,contours[i][hullI[i][j]].y ), cv::Scalar(0, 0, 255),10);
			}
		
			*/
	  }

	  Mat drawing = Mat::zeros( skin.size(), CV_8UC3 );
	
	  for( int i = 0; i< contours.size(); i++ )
	  {
		 
		//  printf("Defects %d\n",defect[i].size());
		  Scalar color = Scalar( 126, 100,89 );
		  drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		  drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	  }

	  
	  namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
	  imshow( "Hull demo", drawing );
	// FindBlobs(fg,blobs);
	// printf("#Blobs %d\n",blobs.size());
	 // GaussianBlur(skin,img,Size(0,0),10);
	
	 //erode(skin,skin,element);
	//test(&frame);
	

//	 showImg("Skin2",fg);
	  showImg("Camera",frame);
	  showImg("Tips",img);
	 if(waitKey(1) == 27)
		return 0;
	}
}

int main()
{
	Mat img,reduced;
	uchar* p;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat lookUpTable(1, 256, CV_8U);
	p = lookUpTable.data;
	for( int i = 0; i < 256; ++i)
		p[i] = 128 * (i/128);
	
	img = imread("C:\\Users\\walkout\\Pictures\\Penguins.jpg",1);

		LUT(img, lookUpTable, img);
	showImg("We",img);
	//showImg("Reduced",reduced);
	cvtColor(img,img,CV_BGR2GRAY);
	 adaptiveThreshold(img,img,70,ADAPTIVE_THRESH_GAUSSIAN_C ,THRESH_BINARY,9, 7);
	//Canny(img,img,80,80);
	showImg("Canny",img);
	
	findContours( img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


	Mat drawing = Mat::zeros( img.size(), CV_8UC3 );

	for( int i = 0; i< contours.size(); i++ )
	{

		Scalar color = Scalar( 126, 100,89 );
		if( contours[i].size()>100 
			&& contours[i][0].x < contours[i][contours[i].size()-1].x +5 
			&& contours[i][0].x > contours[i][contours[i].size()-1].x -5 
			&& contours[i][0].y < contours[i][contours[i].size()-1].y +5 
			&& contours[i][0].y > contours[i][contours[i].size()-1].y -5)
		{
		 drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		 printf("(%d,%d) (%d,%d)\n",contours[i][0].x,contours[i][0].y,contours[i][contours[i].size()-1].x,contours[i][contours[i].size()-1].y);
		 waitKey(10);
		 showImg("Cont",drawing);
		}
	}

	showImg("Cont",drawing);

	//hand();
	//showImg("A",img);
	printf("ok\n");
	waitKey(0);


	return 0;

}