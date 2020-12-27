#ifndef ETF_H
#define ETF_H

#include <opencv2/core.hpp>

struct ETF
{
	cv::Mat tx;
	cv::Mat ty;
	cv::Mat mag;

	ETF(const cv::Mat& picGray);
	ETF(const ETF& oth);
	ETF& operator=(const ETF& oth);
	void set(const cv::Mat& picGray);
	void smooth(const int HalfW, const int M);
};

#endif // !ETF_H
