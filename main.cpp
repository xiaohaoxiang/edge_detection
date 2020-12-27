#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include "defs.h"
#include "prewitt.h"
#include "sobel.h"
#include "canny.h"
#include "fdog.h"

using std::cout;
using std::endl;

struct
{
	std::string name;
	std::function<void(const cv::Mat&, cv::Mat&)> func;
}
Functions[]
{
	{"Prewitt", getPrewitt},
	{"Sobel",   getSobel},
	{"Canny",   getCanny},
	{"FDoG",    getFDoG}
};

int main()
{
	std::ifstream picNameStream("pictures.txt");
	std::string picName;
	for (cv::Mat picRaw, picGray, picFilter, edge[4]; std::getline(picNameStream, picName);)
	{
		cout << picName << endl;
		picRaw = cv::imread(picName);
		if (picRaw.empty()) continue;
		cv::cvtColor(picRaw, picGray, cv::COLOR_RGB2GRAY);
		cv::bilateralFilter(picGray, picFilter, 15, 15, 10);
		picGray = picFilter;

		auto run = [&](cv::Mat& dst, const std::string& name, std::function<void(const cv::Mat&, cv::Mat&)> func)
		{
			cout << "start " << name << endl;
			auto t0 = std::chrono::steady_clock::now();
			func(picGray, dst);
			cout << "finish " << name << " in "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() << "ms\n" << endl;
		};

		const int n = sizeof(Functions) / sizeof(*Functions);
		for (int i = 0; i < n; i++)
			run(edge[i], Functions[i].name, Functions[i].func);
		for (int i = 0; i < n; i++)
		{
			cv::imshow(Functions[i].name, edge[i]);
			cv::moveWindow(Functions[i].name, i * (picRaw.cols / 2), picRaw.rows + 36 * (i + 1));
		}
		cv::imshow("Raw Picture", picRaw);
		cv::moveWindow("Raw Picture", 0, 0);
		cv::imshow("Gray Picture", picGray);
		cv::moveWindow("Gray Picture", picRaw.cols + 1, 0);

		if (cv::waitKey() == 'q')
		{
			break;
		}
		cv::destroyAllWindows();
	}

	return 0;
}
