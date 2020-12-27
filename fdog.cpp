#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include "etf.h"

using std::abs;
using std::min;
using std::max;
using std::round;

static const double Pi = std::acos(-1);

inline double gauss(double x, double mean, double sigma)
{
	return (std::exp((-(x - mean) * (x - mean)) / (2 * sigma * sigma)) / std::sqrt(Pi * 2.0 * sigma * sigma));
}

std::vector<double> makeGaussianVector(const double sigma)
{
	static const double thres = 1e-3;

	std::vector<double> gau;
	for (int i=0;;i++)
	{
		double g = gauss(double(i), 0.0, sigma);
		gau.push_back(g);
		if (g < thres)
		{
			break;
		}
	}
	return gau;
}

void getDirectionalDoG(const cv::Mat& image, cv::Mat& dog, const ETF& e, const std::vector<double>& gau1, const std::vector<double>& gau2, const double tau)
{
	const int halfW1 = int(gau1.size() - 1), halfW2 = int(gau2.size() - 1);

	dog = cv::Mat(image.size(), CV_64F);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			double sum1 = 0, sum2 = 0;
			double wSum1 = 0, wSum2 = 0;
			double vn[2]{ -e.ty.at<double>(i, j), e.tx.at<double>(i, j) };

			if (vn[0] == 0 && vn[1] == 0)
			{
				sum1 = 255;
				sum2 = 255;
				dog.at<double>(i, j) = sum1 - tau * sum2;
				continue;
			}
			double d_x = i, d_y = j;
			for (int s = -halfW2; s <= halfW2; s++)
			{
				double x = d_x + vn[0] * s, y = d_y + vn[1] * s;
				if (x < 0 || y < 0 || x >= image.rows || y >= image.cols)
					continue;
				int xi = int(round(min(max(x, 0.0), image.rows - 1.0))),
					yi = int(round(min(max(y, 0.0), image.cols - 1.0)));
				double val = image.at<uchar>(xi, yi);
				int dd = abs(s);
				double weight1 = (dd < halfW1 ? gau1[dd] : 0), weight2 = gau2[dd];
				sum1 += val * weight1;
				wSum1 += weight1;
				sum2 += val * weight2;
				wSum2 += weight2;
			}
			sum1 /= wSum1;
			sum2 /= wSum2;
			dog.at<double>(i, j) = sum1 - tau * sum2;
		}
	}
}

void getFlowDoG(cv::Mat& dog, const ETF& e, const std::vector<double>& gau3)
{
	cv::Mat tmp(dog.size(), CV_64F);
	int halfL = int(gau3.size() - 1);
	const double stepSize = 1;

	for (int i = 0; i < dog.rows; i++)
	{
		for (int j = 0; j < dog.cols; j++)
		{
			double
				weight1 = gau3[0],
				wSum1 = weight1,
				val = dog.at<double>(i, j),
				sum1 = val * weight1;
			double d_x = i, d_y = j;
			int i_x = i, i_y = j;
			for (int k = 1; k < halfL; k++)
			{
				double vt[2]{ e.tx.at<double>(i_x, i_y), e.ty.at<double>(i_x, i_y) };
				if (vt[0] == 0 && vt[1] == 0)
					break;
				double x = d_x, y = d_y;
				if (x < 0 || y < 0 || x >= dog.rows || y >= dog.cols)
					break;
				int xi = int(round(min(max(x, 0.0), dog.rows - 1.0))),
					yi = int(round(min(max(y, 0.0), dog.cols - 1.0)));
				val = dog.at<double>(xi, yi);
				weight1 = gau3[k];
				sum1 += val * weight1;
				wSum1 += weight1;
				d_x += vt[0] * stepSize;
				d_y += vt[1] * stepSize;
				i_x = int(round(d_x));
				i_y = int(round(d_y));
				if (i_x < 0 || i_y < 0 || i_x >= dog.rows || i_y >= dog.cols)
					break;
			}
			d_x = double(i);
			d_y = double(j);
			i_x = i;
			i_y = j;
			for (int k = 1; k < halfL; k++)
			{
				double vt[2]{ -e.tx.at<double>(i_x, i_y) ,-e.ty.at<double>(i_x,i_y) };
				if (vt[0] == 0 && vt[1] == 0)
					break;
				double x = d_x, y = d_y;
				if (x < 0 || y < 0 || x >= dog.rows || y >= dog.cols)
					break;
				int xi = int(round(min(max(x, 0.0), dog.rows - 1.0))),
					yi = int(round(min(max(y, 0.0), dog.cols - 1.0)));
				val = dog.at<double>(xi, yi);
				weight1 = gau3[k];
				sum1 += val * weight1;
				wSum1 += weight1;
				d_x += vt[0] * stepSize;
				d_y += vt[1] * stepSize;
				i_x = int(round(d_x));
				i_y = int(round(d_y));
				if (i_x < 0 || i_y < 0 || i_x >= dog.rows || i_y >= dog.cols)
					break;
			}
			sum1 /= wSum1;
			
			if (sum1 > 0) tmp.at<double>(i, j) = 1;
			else tmp.at<double>(i, j) = 1 + std::tanh(sum1);
		}
	}
	dog = cv::Mat(dog.size(), CV_8U);
	for (int i = 0; i < dog.rows; i++)
	{
		for (int j = 0; j < dog.cols; j++)
		{
			double v = round(tmp.at<double>(i, j) * 255);
			dog.at<uchar>(i, j) = uchar(v > 255 ? 255 : v);
		}
	}
}

void getFDoG(const cv::Mat& picGray, cv::Mat& picFDoG, const ETF& e, const double sigma, const double sigma3, const double tau)
{
	static const int iterFDoG = 3;

	cv::Mat tmp;
	picGray.copyTo(tmp);
	for (int i = 0; i < iterFDoG; i++, cv::addWeighted(tmp, 1, picFDoG, 1, 0, tmp))
	{
		getDirectionalDoG(tmp, picFDoG, e, makeGaussianVector(sigma), makeGaussianVector(1.6 * sigma), tau);
		getFlowDoG(picFDoG, e, makeGaussianVector(sigma3));
	}
}

void getFDoG(const cv::Mat& picGray, cv::Mat& picFDoG)
{
	ETF e(picGray);
	e.smooth(4, 2);
	return getFDoG(picGray, picFDoG, e, 1, 3, 0.99);
}
