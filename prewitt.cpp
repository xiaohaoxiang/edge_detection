#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void getPrewitt(const cv::Mat& picGray, cv::Mat& picPrewitt)
{
	cv::Mat picPrewittX, picPrewittY, kx = (cv::Mat_<short>(1, 3) << 1, 1, 1), ky = (cv::Mat_<short>(1, 3) << -1, 0, 1);
	cv::sepFilter2D(picGray, picPrewittX, CV_16S, kx, ky);
	cv::sepFilter2D(picGray, picPrewittY, CV_16S, ky, kx);
	cv::convertScaleAbs(picPrewittX, picPrewittX);
	cv::convertScaleAbs(picPrewittY, picPrewittY);
	picPrewitt = cv::max(picPrewittX, picPrewittY);
	cv::bitwise_not(picPrewitt, picPrewitt);
}
