#ifndef SOBEL_H
#define SOBEL_H

#include <opencv2/core.hpp>

void getSobel(const cv::Mat& picGray, cv::Mat& picSobel);

#endif // !SOBEL_H
