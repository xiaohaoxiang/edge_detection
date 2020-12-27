#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include "etf.h"

using std::abs;
using std::min;
using std::max;

static const double MaxVal = 1020;

inline void make_unit(double& vx, double& vy)
{
	double mag = std::sqrt(vx * vx + vy * vy);
	if (mag != 0.0)
	{
		vx /= mag;
		vy /= mag;
	}
}

ETF::ETF(const cv::Mat& picGray)
{
	set(picGray);
}

ETF::ETF(const ETF& oth)
{
	*this = oth;
}

ETF& ETF::operator=(const ETF& oth)
{
	oth.tx.copyTo(tx);
	oth.ty.copyTo(ty);
	oth.mag.copyTo(mag);
	return *this;
}

void ETF::set(const cv::Mat& picGray)
{
	mag = cv::Mat(picGray.size(), CV_64F);

	cv::Sobel(picGray, tx, CV_16S, 1, 0);
	cv::Sobel(picGray, ty, CV_16S, 0, 1);
	tx.convertTo(tx, CV_64F, -1.0 / MaxVal);
	ty.convertTo(ty, CV_64F, 1.0 / MaxVal);

	for (int i = 0; i < picGray.rows; i++)
	{
		for (int j = 0; j < picGray.cols; j++)
		{
			double& vx = tx.at<double>(i, j), & vy = ty.at<double>(i, j), & cur = mag.at<double>(i, j);
			cur = std::sqrt(vx * vx + vy * vy);
			if (cur != 0.0)
			{
				vx /= cur;
				vy /= cur;
			}
		}
	}
	double maxGrad;
	cv::minMaxLoc(mag, nullptr, &maxGrad);
	mag *= 1.0 / maxGrad;
}

void ETF::smooth(const int HalfW, const int M)
{
	ETF e2(*this);
	for (int k = 0; k < M; k++)
	{
		for (int dir = 0; dir < 2; dir++)
		{
			for (int j = 0; j < mag.cols; j++)
			{
				for (int i = 0; i < mag.rows; i++)
				{
					double g[2]{}, v[2]{ tx.at<double>(i, j), ty.at<double>(i, j) };
					for (int s = -HalfW; s <= HalfW; s++)
					{
						int x, y;
						if (dir)
							x = i, y = min(max(j + s, 0), mag.cols - 1);
						else
							x = min(max(i + s, 0), mag.rows - 1), y = j;
						double magDiff = mag.at<double>(x, y) - mag.at<double>(i, j),
							w[2]{ tx.at<double>(x, y), ty.at<double>(x, y) };
						double angle = v[0] * w[0] + v[1] * w[1];
						double factor = angle < 0 ? -1 : 1;
						double weight = magDiff + 1;

						g[0] += weight * w[0] * factor;
						g[1] += weight * w[1] * factor;
					}
					make_unit(g[0], g[1]);
					e2.tx.at<double>(i, j) = g[0];
					e2.ty.at<double>(i, j) = g[1];
				}
			}
			*this = e2;
		}
	}
}
