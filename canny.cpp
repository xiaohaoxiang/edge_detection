#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void getCanny(const cv::Mat& picGray, cv::Mat& picCanny)
{
	cv::Canny(picGray, picCanny, 128, 160);
	cv::bitwise_not(picCanny, picCanny);
}
