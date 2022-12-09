// filter.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "opencv2/opencv.hpp"
#include<algorithm>
using namespace std;
using namespace cv;

Mat dst;
Mat a;
Mat t,tt;
int bright= 50, contrast = 50, saturation = 50, coltemp = 50, hlight = 50, gama_v = 50;
Mat s_mean;
double m;
vector<pair<int, int>>color_hue[6];

int colrate[6] = { 0 }, hue_sum_range[6] = { 0 }, hue_sat[6] = { 0 }, hue_lig[6] = { 0 };
uchar cal_new_h( int ori, pair<int,int> range, pair<int,int> shade, double rate,int red_flag = 0 ) {
	int lrange = range.first, rrange = range.second;
	int lshade = shade.first, rshade = shade.second;
	if (red_flag) {
		if (lshade <= ori && ori <= 179 || 0 <= ori && ori <= rshade) {
			ori = (ori + (int)((rshade + 180 - lshade + 1) * rate) + 180) % 180;
		}
		else if (lrange <= ori && ori <= lshade) {
			ori = (ori + (int)((lshade - lrange + 1) * rate) + 180) % 180;
		}
		else if (rshade <= ori && ori <= rrange) {
			ori = (ori + (int)((rrange - rshade + 1) * rate) + 180) % 180;
		}
	}
	else {
		if (lshade <= ori && ori <= rshade) {
			ori = (ori + (int)((rshade - lshade + 1) * rate) + 180) % 180;
		}
		else if (lrange <= ori && ori <= lshade) {
			ori = (ori + (int)((lshade - lrange + 1) * rate) + 180) % 180;
		}
		else if (rshade <= ori && ori <= rrange) {
			ori = (ori + (int)((rrange - rshade + 1) * rate) + 180) % 180;
		}
	}
	return ori;
}
uchar cal_new_s(int hue, int ori, pair<int, int> range, pair<int, int> shade, double rate, int red_flag = 0) {
	int lrange = range.first, rrange = range.second;
	int lshade = shade.first, rshade = shade.second;
	if (red_flag) {
		if (lshade <= hue && hue <= 179 || 0 <= hue && hue <= rshade) {
			ori = saturate_cast<uchar>(ori + ori * rate);
		}
		else if (lrange <= hue && hue <= lshade) {
			double nrate = (1.0 * hue - lrange) / (lshade - lrange);
			ori = saturate_cast<uchar>(ori + ori * rate * nrate);
		}
		else if (rshade <= hue && hue <= rrange) {
			double nrate = (1.0 * rrange - hue) / (rrange - rshade);
			ori = saturate_cast<uchar>(ori + ori * rate * nrate);
		}
	}
	else {
		if (lshade <= hue && hue <= rshade) {
			ori = saturate_cast<uchar>(ori  + ori * rate) ;
		}
		else if (lrange <= hue && hue <= lshade) {
			double nrate = (1.0 * hue - lrange) / (lshade - lrange);
			ori = saturate_cast<uchar>(ori  + ori * rate * nrate) ;
		}
		else if (rshade <= hue && hue <= rrange) {
			double nrate = (1.0 * rrange - hue) / (rrange - rshade);
			ori = saturate_cast<uchar>(ori  + ori * rate * nrate)  ;
		}
	}
	return ori < 0 ? 0 : ori;
}

uchar cal_new_l(int hue, int sat, int lig, pair<int, int> range,int mid, double rate, int red_flag = 0) {
	int lrange = range.first;
	int rrange = range.second;
	float sat_rate = sat / 255.0;
	if (red_flag) {
		float lsum = 179 - lrange + 1;
		float rsum = rrange - mid + 1;
		if (hue == mid) {
			lig = saturate_cast<uchar>(lig + 255 * rate * sat_rate) ;
		}
		else if (lrange <= hue) {
			float hue_rate = (hue - lrange) / lsum;
			lig = saturate_cast<uchar>(lig + 255 * rate * hue_rate * sat_rate);
		}
		else {
			float hue_rate = (rrange - hue)/ rsum;
			lig = saturate_cast<uchar>(lig + 255 * rate * hue_rate * sat_rate);
		}
	}
	else {
		float lsum = mid - lrange + 1;
		float rsum = rrange - mid + 1;
		if (hue == mid) {
			lig = saturate_cast<uchar>(lig + 255 * rate * sat_rate);
		}
		else if (hue <= mid) {
			float hue_rate = (hue - lrange) / lsum;
			lig = saturate_cast<uchar>(lig + 255 * rate * hue_rate * sat_rate);
		}
		else {
			float hue_rate = (rrange - hue) / rsum;
			lig = saturate_cast<uchar>(lig + 255 * rate * hue_rate * sat_rate);
		}
	}
	return lig < 0 ? 0 : lig;
}
void init_color_hue(Mat& src) {
	Mat tmp = src.clone();
	cvtColor(tmp, tmp, COLOR_BGR2HSV);
	for (int i = 0; i < tmp.rows; i++) {
		for (int j = 0; j < tmp.cols; j++) {
			uchar &hue = tmp.at<Vec3b>(i, j)[0];
			uchar &sat = tmp.at<Vec3b>(i, j)[1];
			uchar &lig = tmp.at<Vec3b>(i, j)[2];
			if (169 <= hue && hue <= 179 || 0 <= hue && hue <= 14)color_hue[0].push_back({ i, j });
			if (5 <= hue && hue <= 21)color_hue[1].push_back({ i, j });//橙
			if (3 <= hue && hue <= 36)color_hue[2].push_back({ i, j });//黄
			if (41 <= hue && hue <= 85)color_hue[3].push_back({ i, j });//绿
			if (90 <= hue && hue <= 145)color_hue[4].push_back({ i, j });//蓝
			if (140 <= hue && hue <= 174)color_hue[5].push_back({ i, j });//紫
		}
	}
	cvtColor(tmp, tmp, COLOR_HSV2BGR);
}

uchar sat_cast(int x, uchar mn, uchar mx) {
	if (x < mn)return mn;
	if (x > mx)return mx;
	return x;
}

void change_Hue(Mat& dst) {
	cvtColor(dst, dst, COLOR_BGR2HSV);
	//uchar& huee = dst.at<Vec3b>(1, 639)[0];
	for (int k = 0; k < 6; k++) {
		double rate = (1.0 * colrate[k] - 50) / 100 ;
		for (int i = 0; i < color_hue[k].size(); i++) {
			int x = color_hue[k][i].first;
			int y = color_hue[k][i].second;
			uchar& hue = dst.at<Vec3b>(x, y)[0];
			if(k == 0)hue = cal_new_h(hue, {170, 14}, {175,7}, rate, 1);
			if(k == 1)hue = cal_new_h(hue, {0, 25}, {8,25}, rate);
			if(k == 2)hue = cal_new_h(hue, {6, 36}, {15,29}, rate);
			if(k == 3)hue = cal_new_h(hue, {41, 85}, {42,85}, rate);
			if(k == 4)hue = cal_new_h(hue, {85, 145}, {100,135}, rate);
			if(k == 5)hue = cal_new_h(hue, {140, 174}, {145,160}, rate);
		}
	}
	cvtColor(dst, dst, COLOR_HSV2BGR);
	//uchar& huee = dst.at<Vec3b>(1, 639)[0];
	
}

void HueSaturate(Mat& src) {
	//Mat tmp = src.clone();
	cvtColor(src, src, COLOR_BGR2HSV);
	for (int k = 0; k < 6; k++) {
		double rate = (1.0 * hue_sat[k] - 50)/100 ;
		for (int i = 0; i < color_hue[k].size(); i++) {
			int x = color_hue[k][i].first;
			int y = color_hue[k][i].second;
			uchar& hue = src.at<Vec3b>(x, y)[0];
			uchar& sat = src.at<Vec3b>(x, y)[1];
			if (k == 0)sat = cal_new_s(hue, sat, { 170, 12 }, { 175,5 }, rate, 1);
			if (k == 1)sat = cal_new_s(hue, sat, { 4, 21 }, { 10,21 }, rate);
			if (k == 2)sat = cal_new_s(hue, sat, { 4, 36 }, { 15,29 }, rate);
			if (k == 3)sat = cal_new_s(hue, sat, { 41, 85 }, { 42,85 }, rate);
			if (k == 4)sat = cal_new_s(hue, sat, { 85, 145 }, { 100,140 }, rate);
			if (k == 5)sat = cal_new_s(hue, sat, { 140, 174 }, { 145,160 }, rate);
			//sat = saturate_cast<uchar>((sat + sat * rate));
		}
	}
	cvtColor(src, src, COLOR_HSV2BGR);
	//imshow("sat2",tmp);
}

void HueLight(Mat& src) {
	//Mat tmp = src.clone();
	cvtColor(src, src, COLOR_BGR2HSV);
	for (int k = 0; k < 6; k++) {
		double rate = (1.0 * hue_lig[k] - 50) /50;
		for (int i = 0; i < color_hue[k].size(); i++) {
			int x = color_hue[k][i].first;
			int y = color_hue[k][i].second;
			uchar& hue = src.at<Vec3b>(x, y)[0];
			uchar& sat = src.at<Vec3b>(x, y)[1];
			uchar& lig = src.at<Vec3b>(x, y)[2];
			if (k == 0)lig = cal_new_l(hue, sat, lig, { 169, 14 }, 0, rate, 1);
			if (k == 1)lig = cal_new_l(hue, sat, lig, { 4, 21 }, 14, rate);
			if (k == 2)lig = cal_new_l(hue, sat, lig, { 4, 36 }, 22, rate);
			if (k == 3)lig = cal_new_l(hue, sat, lig, { 41, 85 },60, rate);
			if (k == 4)lig = cal_new_l(hue, sat, lig, { 85, 145 },130, rate);
			if (k == 5)lig = cal_new_l(hue, sat, lig, { 140, 174 },150, rate);
			//lig = saturate_cast<uchar>((lig  + lig * rate / 200.));
		}
	}
	cvtColor(src, src, COLOR_HSV2BGR);
	//imshow("sat2",tmp);
}
void oncChange(int, void*) {
	dst = a.clone();
	change_Hue(dst);
	HueSaturate(dst);
	HueLight(dst);
	//GaussianBlur(dst, dst, Size(3, 3), 0, 0);
	imshow("hue", dst);
	Mat mask = Mat::zeros(a.size(), a.type());
	for (int i = 0; i < color_hue[0].size(); i++) {
		int x = color_hue[0][i].first;
		int y = color_hue[0][i].second;
		mask.at<Vec3b>(x, y) = dst.at<Vec3b>(x, y);
		//cout << x << " " << y << ":" << mask.at<Vec3b>(x, y) << "\n";
	}
	//Mat andtm = t & mask;
	//imshow("hue", mask);
	//imshow("hue", dst);
}
void Hue() {
	for (int i = 0; i < 6; i++)colrate[i] = 50;
	for (int i = 0; i < 6; i++)hue_sat[i] = 50;
	for (int i = 0; i < 6; i++)hue_lig[i] = 50;
	namedWindow("hue", WINDOW_AUTOSIZE);
	//createTrackbar("红", "hue", &colrate[0], 100, oncChange, 0);
	//createTrackbar("橙", "hue", &colrate[1], 100, oncChange, 0);
	//createTrackbar("黄", "hue", &colrate[2], 100, oncChange, 0);
	//createTrackbar("绿", "hue", &colrate[3], 100, oncChange, 0);
	//createTrackbar("蓝", "hue", &colrate[4], 100, oncChange, 0);
	//createTrackbar("紫", "hue", &colrate[5], 100, oncChange, 0);
	//createTrackbar("红sat", "hue", &hue_sat[0], 100, oncChange, 0);
	//createTrackbar("橙sat", "hue", &hue_sat[1], 100, oncChange, 0);
	//createTrackbar("黄sat", "hue", &hue_sat[2], 100, oncChange, 0);
	//createTrackbar("绿sat", "hue", &hue_sat[3], 100, oncChange, 0);
	//createTrackbar("蓝sat", "hue", &hue_sat[4], 100, oncChange, 0);
	//createTrackbar("紫sat", "hue", &hue_sat[5], 100, oncChange, 0);
	createTrackbar("红lig", "hue", &hue_lig[0], 100, oncChange, 0);
	createTrackbar("橙lig", "hue", &hue_lig[1], 100, oncChange, 0);
	createTrackbar("黄lig", "hue", &hue_lig[2], 100, oncChange, 0);
	createTrackbar("绿lig", "hue", &hue_lig[3], 100, oncChange, 0);
	createTrackbar("蓝lig", "hue", &hue_lig[4], 100, oncChange, 0);
	createTrackbar("紫lig", "hue", &hue_lig[5], 100, oncChange, 0);
	oncChange(0, NULL);
}

void gama(Mat& src) {
	float rate = -(1.0 * gama_v - 50) / 100 + 1;
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), rate) * 255.0f);
	}
	MatIterator_<Vec3b> it, end;
	for (it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; it++)
	{
		(*it)[0] = lut[((*it)[0])];
		(*it)[1] = lut[((*it)[1])];
		(*it)[2] = lut[((*it)[2])];
	}
}
void Brightness(Mat& src){
	double bri = 1.0 * bright - 50;
	src.convertTo(src, -1, 1, bri);
}
void Contrast(Mat& src) {
	double con =  (1.0 * contrast - 50) / 50 + 1 + (1.0 * gama_v - 50) / 1000;
	src.convertTo(src, -1, con, (1.0 - con) * m);
}
void Saturate(Mat& src) {
	//Mat tmp = src.clone();
	cvtColor(src, src, COLOR_BGR2HLS);
	double sat = (1.0 * saturation - 50) / 50;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(src.at<Vec3b>(i, j)[2] * (1.0 + sat));
		}
	}
	cvtColor(src, src, COLOR_HLS2BGR);
	//imshow("sat2",tmp);
}

void ColorTemp(Mat& src) {
	double ct = (coltemp - 50) / 2;
	for (int i = 0; i < src.rows; i++) {
		uchar* s = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			uchar &r = s[j * 3 + 2];
			uchar &g = s[j * 3 + 1];
			uchar &b = s[j * 3];
			r = saturate_cast<uchar>(r + ct);
			g = saturate_cast<uchar>(g + ct);
			b = saturate_cast<uchar>(b - ct);
		}
	}
}
void HighLight(Mat& src) {
	//cout << src.type();
	int highlight_v = hlight - 50;
	Mat gray = Mat::zeros(src.size(), CV_32FC1);
	Mat dst = src.clone();
	//imshow("k", dst);
	dst.convertTo(dst, CV_32FC3);
	vector<Mat>pics;
	split(dst, pics);
	gray = 0.299 * pics[2] + 0.587 * pics[1] + 0.114 * pics[0];
	gray = gray / 255;
	Mat thresh = gray.mul(gray);
	Scalar t = mean(thresh);
	Mat mask = Mat::zeros(gray.size(), CV_8UC1);
	mask.setTo(255, thresh >= t[0]);
	//imshow("123",mask);
	int mx = 4;
	float bri = highlight_v / 100.f / mx;
	float mid = 1.0f + mx * bri;
	Mat midrate = Mat::zeros(src.size(), CV_32FC1);
	Mat brirate = Mat::zeros(src.size(), CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		uchar* m = mask.ptr<uchar>(i);
		float* th = thresh.ptr<float>(i);
		float* mi = midrate.ptr<float>(i);
		float* br = brirate.ptr<float>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (m[j] == 255)
			{
				mi[j] = mid;
				br[j] = bri;
			}
			else {
				mi[j] = (mid - 1.0) / t[0] * th[j] + 1.0f;
				br[j] = (1.0 / t[0] * th[j]) * bri;
			}
		}
	}

	Mat result = Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
	{
		float* mi = midrate.ptr<float>(i);
		float* br = brirate.ptr<float>(i);
		uchar* in = src.ptr<uchar>(i);
		uchar* r = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				float temp = pow(float(in[3 * j + k]) / 255.f, 1.0f / mi[j]) * (1.0f / (1.0f - br[j]));
				if (temp > 1.0f)temp = 1.0f;
				if (temp < 0.0f)temp = 0.0f;
				uchar utemp = uchar(255 * temp);
				r[3 * j + k] = utemp;
			}
		}
	}
	src = result;
}
void noise(Mat &p) {
	Mat src = p.clone();
	Mat t = Mat::zeros(src.size(), src.type());
	RNG rng;
	rng.fill(t, RNG::NORMAL, 0, 10);
	src += t;
	imshow("t", src);
}
void onbChange(int, void*) {
	//dst = a.clone();
	Contrast(dst);
	//Brightness(dst);
	gama(dst);
	//Bri2(dst);
	Saturate(dst);
	ColorTemp(dst);
	//Mat tmp = a.clone();
	HighLight(dst);
	//noise(dst);
	
	imshow("a", dst);
}

int main()
{
    a = imread("./t3.png");
	m = cv::mean(a, s_mean)[0];
	init_color_hue(a);
	Hue();
	//namedWindow("test",WINDOW_AUTOSIZE);
	//createTrackbar("对比度","test", &contrast, 100, onbChange,0);
	//createTrackbar("饱和度","test", &saturation, 100, onbChange,0);
	//createTrackbar("色温","test", &coltemp, 100, onbChange,0);
	////createTrackbar("亮度","test", &bright, 100, onbChange,0);
	//createTrackbar("高光处理","test", &hlight, 100, onbChange,0);
	//createTrackbar("伽马亮度", "test", &gama_v, 100, onbChange, 0);
	//onbChange(0, NULL);
	
	//cvtColor(a, a, COLOR_BGR2RGB);
	//Brightness(a, 30);
	//Mat b;
	//a.convertTo(a, -1, 0.5, (1-0.5)*127);//对比度
	//a.convertTo(a, -1, 1, -50);
	//gama(a, 0.9);
	//cvtColor(a, a, COLOR_RGB2BGR);
	//Contrast(a, 100);
	//imshow("1", a);
    waitKey(0);
}
