#ifndef FDOG_H
#define FDOG_H

#include <opencv2/core.hpp>
#include "etf.h"

void getFDoG(const cv::Mat& picGray, cv::Mat& picFDoG);

#endif // !FDOG_H
