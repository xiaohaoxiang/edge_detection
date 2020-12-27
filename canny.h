#ifndef CANNY_H
#define CANNY_H

#include <opencv2/core.hpp>

void getCanny(const cv::Mat& picGray, cv::Mat& picCanny);

#endif // !CANNY_H
