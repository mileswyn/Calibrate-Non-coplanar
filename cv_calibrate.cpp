#include <opencv.hpp>
#include<core\core.hpp>
#include<highgui\highgui.hpp>
#include<opencv.hpp>
#include <algorithm>
#include <functional>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cv.h>
#include <math.h>

using namespace std;
using namespace cv;


//下面四行用来产生随机序列
// random generator function:
ptrdiff_t myrandom(ptrdiff_t i) { return rand() % i; }
// pointer object to it:
ptrdiff_t(*p_myrandom)(ptrdiff_t) = myrandom;

//以下四个函数全部对应于py中calibrate_dlt的内容
cv::Mat calibrateDLTA(cv::Mat, cv::Mat, cv::Mat&, cv::Mat&, vector<int>&, int, int);
cv::Mat calibrateDLTB(cv::Mat, cv::Mat&, vector<int>&, int, int);
cv::Mat lstsq(cv::Mat, cv::Mat, cv::Mat&); //用来进行最小二乘
cv::Mat lstsq_aux(cv::Mat, cv::Mat, cv::Mat&); //将最小二乘得到的向量进行reshape
//以下对应于intric_mat的内容
cv::Mat intrin_mat(cv::Mat);
//以下对应calibrate_zzy的内容
void calibrate_zzy(cv::Mat, cv::Mat, cv::Mat, cv::Mat, vector<int>&, int, int);

static cv::Mat calibrateDLTA(cv::Mat W, cv::Mat P, cv::Mat& A, cv::Mat& At, vector<int>& cal, int imgheight, int imgwidth)
{
	float simpleArray1[5] = { 1,0,0,0,0 };
	cv::Mat Sarray1 = cv::Mat(1, 5, CV_32FC1, simpleArray1);
	cv::Mat Sarray2 = cv::Mat(1, 4, CV_32FC1, { 0,0,0,0 });
	cv::Mat Sarray3 = cv::Mat(1, 1, CV_32FC1, { 1 });
	cv::Mat tmp2 = cv::Mat(1, 1, CV_32FC1);
	//tmp2 = -P.at<Vec2f>(2, 0)[1];
	//cout << "看看tmp2" << tmp2 << endl;
	for (int i = 0; i < 8; ++i)
	{
		int iflag = cal[i];
		cv::Mat temp1, temp2, temp3, temp4, Atemp1, Atemp2;
		cv::hconcat(W.row(iflag).clone(), Sarray1, temp1);
		cv::hconcat(temp1, -P.at<Vec2f>(i, 0)[0] * W.row(iflag).clone(), Atemp1);
		cv::hconcat(Sarray2, W.row(iflag).clone(), temp3);
		cv::hconcat(temp3, Sarray3, temp4);
		cv::hconcat(temp4, -P.at<Vec2f>(i, 0)[1] * W.row(iflag).clone(), Atemp2);
		if (i == 0)
		{
			At = Atemp1;
		}
		else
		{
			cv::vconcat(At, Atemp1, At);
			/*out << "第i次循环At" << At << endl;*/
		}
		cv::vconcat(At, Atemp2, At);
		if (i == 7)
		{
			A = At;
		}
		else
			continue;
		//if (i == 7) cout << At << endl;
	}
	return A;
}

static cv::Mat calibrateDLTB(cv::Mat P, cv::Mat& B, vector<int>& cal, int imgheight, int imgwidth)
{
	for (int i = 0; i < 8; ++i)
	{
		int iflag = cal[i];
		if (i == 0)
		{
			B = P.row(iflag);
		}
		else
		{
			cv::hconcat(B, P.row(iflag), B);
			/*out << "第i次循环At" << At << endl;*/
		}
	}
	return B.t();
}

static cv::Mat lstsq(cv::Mat A, cv::Mat B, cv::Mat& L)
{
	cv::solve(A, B, L, CV_SVD);
	//CV_LU - 最佳主元选取的高斯消除法(这个好像用不了？因为无法直接用求逆计算？)
	//CV_SVD - 奇异值分解法
	//CV_SVD_SYM - 对正定对称矩阵的SVD方法
	return L;
}

static cv::Mat lstsq_aux(cv::Mat A, cv::Mat B, cv::Mat& L_norm)
{
	cv::Mat aux1 = cv::Mat(1, 1, CV_32FC1, { 1 });
	cv::solve(A, B, L_norm, CV_SVD);
	cv::vconcat(L_norm, aux1, L_norm);
	L_norm = L_norm.reshape(0, 3); //注意opencv中reshape的第一个参数是通道数，0为保持不变
	return L_norm;
}

static cv::Mat intrin_mat(cv::Mat M)
{
	float m34,u0,v0;
	cv::Mat ax_aux(cv::Size(1, 3), CV_32FC1), ay_aux(cv::Size(1, 3), CV_32FC1), ax, ay;//前两个定义必须要定义大小，否则下面输出大小为空
	cv::Mat m1(cv::Size(1, 3), CV_32FC1);
	cv::Mat m2(cv::Size(1, 3), CV_32FC1);
	cv::Mat m3(cv::Size(1, 3), CV_32FC1);
	m1 = M.row(0).colRange(0, 3).clone();
	m2 = M.row(1).colRange(0, 3).clone();
	m3 = M.row(2).colRange(0, 3).clone();
	m34 = 1 / sqrt(m3.dot(m3));
	u0 = m34 * m34*(m1.dot(m3));
	v0 = m34 * m34*(m2.dot(m3));
	ax_aux = norm(m1.cross(m3)); // 这里写的不是很好，在cvNorm（用不了）和norm之间用norm求出一个矩阵
	//cout << ax_aux.col(0).row(0).clone() << endl;
	ax = m34 * m34 * ax_aux.col(0).row(0).clone(); //本来想把ax写为一个double，结果不会提取mat中元素，于是定义为mat
	ay_aux = norm(m2.cross(m3));
	ay = m34 * m34 * ay_aux.col(0).row(0).clone();

	cv::Mat K=cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	ax.col(0).row(0).copyTo(K.col(0).row(0));
	ay.col(0).row(0).copyTo(K.col(1).row(1));//将mat元素赋给mat
	K.row(0).col(2) = u0;
	K.row(1).col(2) = v0;
	K.row(2).col(2) = 1;
	return K;
}

static void calibrate_zzy(cv::Mat obj_point, cv::Mat img_point1, cv::Mat img_point2, cv::Mat Ax, vector<int>& cal, int imgheight, int imgwidth)
{
	Size imageSize;
	imageSize.width = imgwidth;
	imageSize.height = imgheight;
	float dist[5] = { 0,0,0,0,0 };
	cv::Mat cameraMatrix= cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	cv::Mat distCoeffs = cv::Mat(1, 5, CV_32FC1, dist);
	cv::Mat Sarray2 = cv::Mat(1, 4, CV_32FC1, { 0,0,0,0 });
	cv::Mat W_shuf(cv::Size(3, 8), CV_32FC1);
	cv::Mat P1_shuf(cv::Size(2, 8), CV_32FC1);
	cv::Mat P2_shuf(cv::Size(2, 8), CV_32FC1);
	vector<Mat> rvecsMat;  // 存放所有图像的旋转向量，每一副图像的旋转向量为一个mat
	vector<Mat> tvecsMat;  // 存放所有图像的平移向量，每一副图像的平移向量为一个mat
	for (int i = 0; i < 8; ++i)
	{
		int iflag = cal[i];
		obj_point.row(iflag).copyTo(W_shuf.row(i));
		img_point1.row(iflag).copyTo(P1_shuf.row(i));
		img_point2.row(iflag).copyTo(P2_shuf.row(i));
	}
	Ax.row(0).col(0).copyTo(cameraMatrix.row(0).col(0));
	Ax.row(0).col(0).copyTo(cameraMatrix.row(1).col(1));
	cameraMatrix.row(2).col(2) = 1;
	cameraMatrix.row(0).col(2) = imgwidth / 2;
	cameraMatrix.row(1).col(2) = imgheight / 2;
	vector<vector<Point2f>> image_points_seq;
	for (int t = 0; t < 2; t++)
	{
		if (t == 0)
		{
			vector<Point2f> image_points;
			for (int t1 = 0; t1 < 8; t1++)
			{
				Point2f imgPoint;
				imgPoint.x = *(float*)(P1_shuf.ptr<float>(t1) + 0);
				imgPoint.y = *(float*)(P1_shuf.ptr<float>(t1) + 1);
				image_points.push_back(imgPoint);
			}
			image_points_seq.push_back(image_points);
		}
		else
		{
			vector<Point2f> image_points;
			for (int t1 = 0; t1 < 8; t1++)
			{
				Point2f imgPoint;
				imgPoint.x = *(float*)(P2_shuf.ptr<float>(t1) + 0);
				imgPoint.y = *(float*)(P2_shuf.ptr<float>(t1) + 1);
				image_points.push_back(imgPoint);
			}
			image_points_seq.push_back(image_points);
		}
	}
	vector<vector<Point3f>> obj_points_seq;
	for (int j = 0; j < 2; j++)
	{
		vector<Point3f> obj_points;
		for (int k = 0; k < 8; k++)
		{
			Point3f realPoint;
			realPoint.x = *(float*)(W_shuf.ptr<float>(k) + 0);
			realPoint.y = *(float*)(W_shuf.ptr<float>(k) + 1);
			realPoint.z = *(float*)(W_shuf.ptr<float>(k) + 2);
			obj_points.push_back(realPoint);
		}
		obj_points_seq.push_back(obj_points);
	}
	int flag = CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_ASPECT_RATIO | CALIB_FIX_S1_S2_S3_S4\
		| CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5\
		| CV_CALIB_FIX_K6 | CV_CALIB_ZERO_TANGENT_DIST;
	calibrateCamera(obj_points_seq, image_points_seq, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, flag);
	cout << W_shuf << endl;
	cout << cameraMatrix << endl;
	cout << "以下为定标结果---------" << endl;
	cout << "相机内参数矩阵：" << cameraMatrix << endl;
	cout << "畸变系数：" << distCoeffs << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	for (int p = 0; p < 2; p++)
	{
		cout << "第" << p + 1 << "幅图像的旋转向量" << endl;
		cout << rvecsMat[p] << endl;
		//将旋转向量转换为对应旋转矩阵
		Rodrigues(rvecsMat[p], rotation_matrix);
		cout << "第" << p + 1 << "幅图像的旋转矩阵" << endl;
		cout << rotation_matrix << endl;
		cout << "第" << p + 1 << "幅图像的平移向量" << endl;
		cout << tvecsMat[p] << endl;
	}
}


int main()
{
	// 将世界坐标系下的点存入矩阵
	float w[10][3] = { {0, 0, 0},{0, 0, 21},{32.5, 0, 14.5},{26.5, 0, 11.5},{53, 0, 2},{65, 5.5, 33.5},{9, 5.5, 33.5},{65, 27.5, 33.5},{37, 16.5, 33.5},{22, 26.5, 33.5} };
	cv::Mat W(cv::Size(3, 10), CV_32FC1, w);
	//W = (W * 1000);
	cout << "世界坐标系下10个点：\n " << W << endl;
	cv::Mat W1 = W.row(3).clone(); //取mat中的某一行
	cout << W1 << endl;
	float p1[10][2] = { {849.0,359.0},{848.0,456.0},{610,404},{654,392},{467,330},{367,455},{784,487},{373,336},{578,413},{700,367} };
	float p2[10][2] = { {787.0,355.0},{768.0,456.0},{527,412},{575,399},{376,339},{272,530},{692,547},{270,657},{484,602},{601,664} };
	cv::Mat P1(cv::Size(2, 10), CV_32FC1, p1);
	cv::Mat P2(cv::Size(2, 10), CV_32FC1, p2);
	cout << "像素坐标系下对应点：\n" << P1 << endl;
	int img_width = 1024;
	int img_height = 1024; //定义图像长宽
	
	//划分点集为8个计算和两个验证
	srand(unsigned(time(NULL)));
	int lsarr[10] = { 0,1,2,3,4,5,6,7,8,9 };
	vector<int>ls(lsarr,lsarr+10);
	vector<int>cal;
	//int cal[8];
	//for (int i=0; i<10; ++i) cout << ls[i] << endl;
	//random_shuffle(ls.begin(), ls.end());
	//random_shuffle(ls.begin(), ls.end(), myrandom);
	for (int i = 0; i < 8; ++i) cal.push_back(ls[i]);
	//for (int i = 0; i < 8; ++i) cal[i]=ls[i];
	int test1 = ls[8]; int test2 = ls[9];
	for (int i = 0; i < 8; ++i) cout << cal[i] << endl;
	//cout << "两个测试数据" << test1 << "\t" << test2 << endl;
	//下面四行可以当作一个输出监视器, vector iterator
	//vector<int>::iterator it;
	//cout << "random vector contains:";
	//for (it = cal.begin(); it != cal.end(); ++it)
	//	cout << " " << *it; //指针

	//以下需要对内参矩阵进行初始化
	//float simpleArray1[5] = { 1,0,0,0,0 };
	//cv::Mat Sarray1 = cv::Mat(1, 5, CV_32FC1, simpleArray1);
	//cv::Mat Sarray2 = cv::Mat(1, 4, CV_32FC1, {0,0,0,0});
	//cv::Mat Sarray3 = cv::Mat(1, 1, CV_32FC1, {1});
	//cout << Sarray1 << endl;
	//v:Mat A,B,At;
	//for (int i = 0; i < 8; ++i)
	//{
	//	int iflag = cal[i];
	//	cv:Mat temp1, temp2, temp3, temp4, Atemp1, Atemp2;
	//	cv::hconcat(W.row(iflag).clone(), Sarray1, temp1);
	//	cv::hconcat(temp1, -P1.at<Vec2f>(i, 0)[0] * W.row(iflag).clone(), Atemp1);
	//	cv::hconcat(Sarray2, W.row(iflag).clone(), temp3);
	//	cv::hconcat(temp3, Sarray3, temp4);
	//	cv::hconcat(temp4, -P1.at<Vec2f>(1, 0)[1] * W.row(iflag).clone(), Atemp2);
	//	if (i == 0)
	//	{
	//		At = Atemp1;
	//		B = P1.row(iflag);
	//	}
	//	else
	//	{
	//		cv::vconcat(At, Atemp1, At);
	//		cv::vconcat(At, Atemp2, At);
	//		cv::hconcat(B, P1.row(iflag), B);
	//		/*out << "第i次循环At" << At << endl;*/
	//	}
	//	if (i == 7)
	//	{
	//		A = At;
	//	}
	//	else 
	//		continue;
	//	//if (i == 7) cout << At << endl;
	//}
	cv::Mat Aout1, A1, At1, Bout1, B1; 
	cv::Mat Aout2, A2, At2, Bout2, B2;
	//py中取了随机
	Aout1 = calibrateDLTA(W, P1, A1, At1, cal, img_width, img_height);
	Bout1 = calibrateDLTB(P1, B1, cal, img_width, img_height);
	Aout2 = calibrateDLTA(W, P2, A2, At2, cal, img_width, img_height);
	Bout2 = calibrateDLTB(P2, B2, cal, img_width, img_height);
	//下面求解超定方程组的最小二乘解
	cv::Mat L1, L2, L_norm1(cv::Size(3, 4), CV_32FC1), L_norm2(cv::Size(3, 4), CV_32FC1);
	L1= lstsq(Aout1, Bout1, L1); //
	L_norm1 = lstsq_aux(Aout1, Bout1, L_norm1);//用DLT方法求出的投影矩阵
	L2 = lstsq(Aout2, Bout2, L2); 
	L_norm2 = lstsq_aux(Aout2, Bout2, L_norm2);
	cout << "reshape之后的L阵为" << L_norm1 << endl;
	//用zzy方法求初始化内参，对应于python代码中的intric_mat
	//暂时跳过了py中的proj函数
	cv::Mat IM1, IM2;
	IM1 = intrin_mat(L_norm1);//此为初始化的内参
	IM2 = intrin_mat(L_norm2);
	cout << IM1 << endl;
	////以下五行用于进行norm的测试
	//cv::Mat tep1 = (Mat_<double>(1,3) << -7.23114734, -0.34214237, -0.757358);
	//cv::Mat tep2 = (Mat_<double>(1,3) << -7.95186965*pow(10,-5), -1.71258385*pow(10, -3), -7.84851474*pow(10, -4));
	//IM1 = norm(tep1.cross(tep2));
	//IM2 = IM1.col(0).row(0).clone();
	//cout << "tttt" << IM1 << endl;

	//下面用张正友方法标定
	cv::Mat ax(cv::Size(1, 1), CV_32FC1);
	ax = (IM1.row(0).col(0).clone() + IM1.row(1).col(1).clone())/2;
	calibrate_zzy(W, P1, P2, ax, cal, img_height, img_width);

	cv::Mat initK;
	//initK = initCamaraMatrix(W, P1, cal, img_height, img_width);
	//cout << "随机序列的前八个值：" << cal;
	//cv::Mat b = cv::Mat(cv::Size(5, 5), CV_32FC1); //3通道每个矩阵元素包含3个uchar值
	//cout << "w  = " << endl << w << endl << endl;
	//cout << "b  = " << endl << b << endl << endl;
	system("pause");
	//VideoCapture capture(0);
	//while (1)
	//{
	//	mat frame;								//定义一个mat变量，用于存储每一帧的图像
	//	capture >> frame;						//读取当前帧    
	//	resize(frame, frame, size(360, 240));	//改变图像大小
	//	imshow("aa", frame);
	//	waitkey(30);							//延时30ms
	//}
	return 0;
}