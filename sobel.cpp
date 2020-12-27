#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void getSobel(const cv::Mat& picGray, cv::Mat& picSobel)
{
	cv::Mat picSobelX, picSobelY;
	cv::Sobel(picGray, picSobelX, CV_64F, 1, 0);
	cv::Sobel(picGray, picSobelY, CV_64F, 0, 1);
	cv::pow(picSobelX, 2, picSobelX);
	cv::pow(picSobelY, 2, picSobelY);
	picSobel = picSobelX + picSobelY;
	cv::sqrt(picSobel, picSobel);
	cv::convertScaleAbs(picSobel, picSobel);
	cv::bitwise_not(picSobel, picSobel);
}
