#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<math.h>


using namespace std;
using namespace cv;

RNG rng(12345);

int g_slider_position = 0;
int g_run = 1, g_dontset = 0;
VideoCapture g_cap;



vector<vector<Point>> find_contours_filter(Mat&frame);
//vector<vector<Point>> Matched_lightbar(vector<vector<Point>>& contours_filter);
vector<RotatedRect> Matched_lightbar_finder(vector<vector<Point>>& contours_filter);
double getDistance(Point pointO, Point pointA);



Point get_topPoint(vector<Point>& contour)
{
    Point Top = *min_element(contour.begin(), contour.end(),
        [](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y; });//将轮廓中点的y值排序，找到最大的y
    return Top;
}

Point get_bottomPoint(vector<Point>& contour)
{
    Point Bottom = *max_element(contour.begin(), contour.end(),
        [](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y; });
    return Bottom;
}

Point get_leftPoint(vector<Point>& contour)
{
    Point Left = *min_element(contour.begin(), contour.end(),
        [](const Point& lhs, const Point& rhs) {return lhs.x < rhs.x; });
    return Left;
}

Point get_rightPoint(vector<Point>& contour)
{
    Point Right = *max_element(contour.begin(), contour.end(),
        [](const Point& lhs, const Point& rhs) {return lhs.x < rhs.x; });
    return Right;
}

void onTrackbarSlide(int pos, void*)
    {
        g_cap.set(CAP_PROP_POS_FRAMES, pos);
        if (!g_dontset)
            g_run = 1;
        g_dontset = 0;
    }


int main()
{
    VideoCapture g_cap("/media/sf_ShareFolder/20exp-2019-5-16-13-27-4.avi");
    Mat frame;
    //roi
    Mat roi;
    Rect roi_rect;
    vector<Rect> roi_vector;
    //获取视频的像素
    int frames = (int)g_cap.get(CAP_PROP_FRAME_COUNT);
    int tmpw = (int)g_cap.get(CAP_PROP_FRAME_WIDTH);
    int tmph = (int)g_cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "Video has " << frames << " frames of dimensions(" << tmpw << "," << tmph << ")" << endl;
    namedWindow("drawing2", WINDOW_NORMAL);
    createTrackbar("Position", "drawing2", &g_slider_position, frames, onTrackbarSlide);
    //设置视频的暂停继续操作
        for (;;)
        {
            if (g_run != 0)
            {
                g_cap >> frame;
                if (frame.empty())break;
                int current_pos = (int)g_cap.get(CAP_PROP_POS_FRAMES);
                g_dontset = 1;
                setTrackbarPos("Position", "drawing2", current_pos);
    //    namedWindow("video", WINDOW_NORMAL);
    //            imshow("video", frame);
                if(current_pos == 1)
                {
                    roi_rect = Rect(0,0,frame.cols,frame.rows);
                    roi_vector.push_back(roi_rect);
                    roi = Mat(frame,roi_vector.at(0));
                }
                vector<vector<Point>> contours_filter = find_contours_filter(roi);//筛选后的灯条
                vector<RotatedRect> Matched_lightbar = Matched_lightbar_finder(contours_filter);
//                vector<vector<Point>> Match = Matched_lightbar(contours_filter);//匹配的灯条
//                vector<RotatedRect> Matched_lightbar;//匹配的旋转矩形
//                for(size_t i = 0; i < Match.size(); i++)
//                {
//                    Matched_lightbar[i] = minAreaRect(Match[i]);
//                }
                vector<RotatedRect> armors;
                Point2f armor_points[4];//装甲板的四个顶点
                Point top1; Point top2; Point bottom1; Point bottom2;

                if(Matched_lightbar.size() == 0)//没有匹配到装甲板
                {
                    roi_rect = Rect(0,0,frame.cols,frame.rows);
                    roi_vector.push_back(roi_rect);
                    cout<<"没有匹配到装甲板"<<endl;
                }

                else//匹配到装甲板
                {
                    for(size_t i = 0; i < Matched_lightbar.size() - 1 ; i += 2)
                    {
//                        top1 = get_topPoint(match[i]);
//                        top2 = get_topPoint(match[i+1]);
//                        bottom1 = get_bottomPoint(match[i]);
//                        bottom1 = get_bottomPoint(match[i+1]);
                        //roi下的坐标
                        Point2f vertices_0[4];
                        Point2f vertices_1[4];
                        Matched_lightbar[i].points(vertices_0);
                        Matched_lightbar[i+1].points(vertices_1);
                        //原图下的坐标
                        Point2f vertices_0_ture[4];
                        Point2f vertices_1_ture[4];
                        //roi后对装甲板的坐标矫正
                        for(int i = 0; i < 4; i++)
                        {
                            vertices_0_ture[i] = Point2f((vertices_0[i].x + roi_vector[current_pos - 1].x),(vertices_0[i].y + roi_vector[current_pos - 1].y));
                        }
                        for(int i = 0; i < 4; i++)
                        {
                            vertices_1_ture[i] = Point2f((vertices_1[i].x + roi_vector[current_pos - 1].x),(vertices_1[i].y + roi_vector[current_pos - 1].y));
                        }
                        Point2f vertices[8];
                        vector<Point> armor_point_8;
                        for(int j = 0; j < 4; j++)
                        {
                            vertices[j] = vertices_0_ture[j];
                        }
                        for(int m = 4; m < 8; m++)
                        {
                            vertices[m] = vertices_1_ture[m-4];
                        }
                        for(int i = 0; i < 8; i++)
                        {
                            armor_point_8.push_back(vertices[i]);
                        }
                        RotatedRect armor = minAreaRect(armor_point_8);
                        armors.push_back(armor);
                        Point2f armor_point_4[4];
                        armor.points(armor_point_4);

                        for(int a = 0; a < 4; a++)
                        {
                            line(frame,vertices_0_ture[a],vertices_0_ture[(a+1) % 4],Scalar(0,255,0),2,8,0);
                        }

                        for(int a = 0; a < 4; a++)
                        {
                            line(frame,vertices_1_ture[a],vertices_1_ture[(a+1) % 4],Scalar(0,255,0),2,8,0);
                        }

                        for(int a = 0; a < 4; a++)
                        {
                            line(frame,armor_point_4[a],armor_point_4[(a+1) % 4],Scalar(0,0,255),1,8,0);
                        }

        //                matching_rect.push_back(point_top[i]);
        //                matching_rect.push_back(point_bottom[i]);
        //                matching_rect.push_back(point_bottom[j]);
        //                matching_rect.push_back(point_top[j]);
        //                polylines(frame, matching_rect, true, Scalar(0, 255, 0), 4);


                    }
                    //roi设置
                    int armor_size = armors.size();
                    cout<<"装甲板的个数: "<<armor_size<<endl;
                    if(armor_size == 1)
                    {
                        Point2f armor_point[4];
                        armors[0].points(armor_point);
                        //将四个点放到另外一个变量中
                        for(int i = 0; i < 4; i++)
                        {
                            armor_points[i] = armor_point[i];
                        }
                        vector<Point> armor_point_vector;
                        for(int i = 0;i < 4; i++)
                        {
                            armor_point_vector.push_back(armor_point[i]);
                        }
                        Rect samll = boundingRect(armor_point_vector);
                        Point tl = samll.tl();
                        Point br = samll.br();
                        Point roi_tl = Point((tl.x)*0.9,(tl.y)*0.9);
                        Point roi_br = Point((br.x)*1.1,(br.y)*1.1);
                        roi_rect = Rect(roi_tl,roi_br);
                        if(roi_rect.x < 0)
                        {
                            roi_rect.x = 0;
                        }
                        if(roi_rect.y < 0)
                        {
                            roi_rect.y = 0;
                        }
                        if(roi_rect.x + roi_rect.width > tmpw)
                        {
                           roi_rect.width = tmpw - roi_rect.x;
                        }
                        if(roi_rect.y + roi_rect.height > tmph)
                        {
                            roi_rect.height = tmph - roi_rect.y;
                        }
                        roi_vector.push_back(roi_rect);
                    }
                    if(armor_size > 1)
                    {
                        double area[armors.size()];
                        for(int i = 0; i < armors.size(); i++)
                        {
                            area[i] = armors[i].size.height*armors[i].size.width;
                        }
                        double biggest_armor_area = *max_element(area, area + armor_size);
                        for(int i = 0; i < armor_size; i++)
                        {
                            if(area[i] == biggest_armor_area)
                            {
                                Point2f armor_point[4];
                                armors[i].points(armor_point);
                                for(int i = 0; i < 4; i++)
                                {
                                    armor_points[i] = armor_point[i];
                                }
                                vector<Point> armor_point_vector;
                                for(int i = 0;i < 4; i++)
                                {
                                    armor_point_vector.push_back(armor_point[i]);
                                }
                                Rect samll = boundingRect(armor_point_vector);
                                Point tl = samll.tl();
                                Point br = samll.br();
                                Point roi_tl = Point((tl.x)*0.9,(tl.y)*0.9);
                                Point roi_br = Point((br.x)*1.1,(br.y)*1.1);
                                roi_rect = Rect(roi_tl,roi_br);
                                if(roi_rect.x < 0)
                                {
                                    roi_rect.x = 0;
                                }
                                if(roi_rect.y < 0)
                                {
                                    roi_rect.y = 0;
                                }
                                if(roi_rect.x + roi_rect.width > tmpw)
                                {
                                   roi_rect.width = tmpw - roi_rect.x;
                                }
                                if(roi_rect.y + roi_rect.height > tmph)
                                {
                                    roi_rect.height = tmph - roi_rect.y;
                                }
                                roi_vector.push_back(roi_rect);
                            }

                        }


                    }
                }
            //roi
            roi = Mat(frame,roi_vector.at(current_pos));
            rectangle(frame,roi_vector.at(current_pos),Scalar(0,255,0),2,8,0);

            //pnp
            top1 = armor_points[1];
            bottom1 = armor_points[0];
            top2 = armor_points[2];
            bottom2 = armor_points[3];

            double h = 28;
            double w = 67;
            //自定义的物体世界坐标，单位为mm
            vector<Point3f> obj=vector<Point3f>
            {
            Point3f(-w, -h, 0),	//tl
            Point3f(w, -h, 0),	//tr
            Point3f(w, h, 0),	//br
            Point3f(-w, h, 0)	//bl
            };
            vector<Point2f> pnts=vector<Point2f>
            {
            Point2f(top1),	    //tl
            Point2f(top2),	    //tr
            Point2f(bottom1),	//br
            Point2f(bottom2)	//bl
            };

            Mat cam = (Mat_<double>(3, 3) << 17029.3, -6.97023 ,680.0940,
                                             0.0000000000000000, 1686.0, 443.3442,
                                             0.0000000000000000, 0.0000000000000000, 1.0000000000000000);
            Mat dis = (Mat_<double>(5, 1) << -0.2134, 1.75150, -0.0470, 4.4455e-04, 0.0000000000000000);

            Mat rVec = Mat::zeros(3, 1, CV_64FC1);//init rvec
            Mat tVec = Mat::zeros(3, 1, CV_64FC1);//init tvec
            //进行位置解算
            solvePnP(obj,pnts,cam,dis,rVec,tVec,false, SOLVEPNP_ITERATIVE);

            double d = sqrt(tVec.at<double>(0)*tVec.at<double>(0)+tVec.at<double>(1)*tVec.at<double>(1)+
                            tVec.at<double>(2)*tVec.at<double>(2));
             Mat rot;
             Rodrigues(rVec, rot);

            string str_d = (string)("distances: ")+ to_string(d);
            string str_r1 = (string)("pitch: ")+ to_string(rot.at<double>(0));//俯仰角
            string str_r2 = (string)("yaw: ")+ to_string(rot.at<double>(1));//偏航角
            string str_r3 = (string)("roll: ")+ to_string(rot.at<double>(2));//翻滚角
            putText(frame, str_d, Point(150, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
            putText(frame, str_r1, Point(10, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
            putText(frame, str_r2, Point(310, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
            putText(frame, str_r3, Point(610, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));



            imshow("drawing2", frame);
            cout << "当前帧数：" << current_pos << "\n" << endl;
            g_run -= 1;

        }

        char c = (char)waitKey(10);
        if (c == 's')//singe step
        {g_run = 1;}
        if (c == 'r')//run mode
        {g_run = -1;}
        if (c == 27)
        {break;}


    }
    return 0;
}


//筛选轮廓
vector<vector<Point>> find_contours_filter(Mat& frame)
{
    Mat frame_gray;
    Mat frame_result;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(frame, frame, MORPH_CLOSE, kernel);
    morphologyEx(frame, frame, MORPH_CLOSE, kernel);
//    namedWindow("frame" , WINDOW_NORMAL);
//    imshow("frame", frame);
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    threshold(frame_gray, frame_gray, 50, 255, THRESH_BINARY);
//    namedWindow("frame_gray" , WINDOW_NORMAL);
//    imshow("frame_gray", frame_gray);
    vector<Mat> splited;
    split(frame, splited);
    Mat splited_result;
    subtract(splited[0], splited[2], splited_result);
    threshold(splited_result, splited_result, 60, 255, THRESH_BINARY);
    dilate(splited_result, splited_result, kernel);
    splited_result = splited_result | splited_result;
//    namedWindow("splited_result , WINDOW_NORMAL);
//    imshow("splited_result", splited_result);
    frame_result = frame_gray & splited_result;
    namedWindow("frame_result", WINDOW_NORMAL);
    imshow("frame_result", frame_result);


    //提取轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(frame_result, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));//检索所有的轮廓并重新建立网状轮廓结构（不同的method会影响hierarchy内部的值）
    vector<Rect>boundRect(contours.size());


//    cout << "矩形个数:" << contours.size() << endl;//获取每帧矩形的个数

    //计算图像矩
    vector<Moments>mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }



    for (size_t i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(Mat(contours[i]));
    }

    Mat drawing = Mat::zeros(frame_gray.size(), CV_8UC3);

    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours, int(i), color, 1, 8, vector<Vec4i>(), 0);
        rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
    }

//    namedWindow("drawing", WINDOW_NORMAL);
//    imshow("drawing", drawing);

    vector<RotatedRect> spinrect(contours.size());//初始化旋转矩形集的大小
    //vector<RotatedRect> spinrect_suit;

    //轮廓的上下顶点
    vector<Point2f> point_top;
    vector<Point2f> point_bottom;


    //定义所需的内参
    double spinrect_ratio[contours.size()];//通过指针可以定义以动态变量位长度的数组
    double lightbar_area[contours.size()];
    double contours_angle[contours.size()];
    vector<vector<Point>> contours_Filter;

    for (size_t i = 0; i < contours.size(); i++)
    {
        spinrect[i] = minAreaRect(contours[i]);//通过轮廓获得符合条件的最小旋转矩形
        lightbar_area[i] = mu[i].m00;//获取灯条的面积



        point_top.push_back(get_topPoint(contours[i]));
        point_bottom.push_back(get_bottomPoint(contours[i]));
        contours_angle[i] = fastAtan2(point_top[i].y - point_bottom[i].y, point_top[i].x - point_bottom[i].x);
        spinrect_ratio[i] = max(spinrect[i].size.height, spinrect[i].size.width) / min(spinrect[i].size.height, spinrect[i].size.width);//获取旋转矩形的长宽比

//        cout << "轮廓面积" << i << ":" << lightbar_area[i] << endl;
        cout << "轮廓角度" << i << ":" << contours_angle[i] << endl;
//        cout << "spin_area " <<i<<":"<< spinrect[i].size.height*spinrect[i].size.width << endl;
//        cout << "长宽比" << i << ":" << spinrect_ratio[i] << endl;

        //筛选适合条件的旋转矩形(粗过滤)
        if (contours.size() > 15) { continue; }//滤过含矩形数量较高的帧
        if (lightbar_area[i] < 50 || lightbar_area[i]>5000) { continue; };//根据面积过滤旋转矩形
        if (spinrect_ratio[i] < 1.5 || spinrect_ratio[i]>15) { continue; }//滤去长宽比过大或者过小的的矩形
        if (contours_angle[i] < 60 | contours_angle[i] > 300) { continue; }
        contours_Filter.push_back(contours[i]);
    }
    cout<<"粗过滤后的灯条个数: "<<contours_Filter.size()<<endl;

    return contours_Filter;
    //获取旋转矩形数据
}


//获取两点之间的距离
double getDistance(Point pointO, Point pointA)

{
    double distance;
    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
    distance = sqrtf(distance);
    return distance;
}

//返回旋转矩形
vector<RotatedRect> Matched_lightbar_finder(vector<vector<Point>>& contours_filter)
{

    vector<RotatedRect> spinrect;//初始的旋转矩形
    vector<RotatedRect> matched_spinrect;//匹配完成的旋转矩形
//    vector<vector<Point>> match;//匹配完成的灯条
    for (size_t i = 0; i < contours_filter.size(); i++)
    {
        spinrect.push_back(minAreaRect(contours_filter[i]));
    }


    int spinrect_size = spinrect.size();

    if (spinrect_size < 2)
    {
        cout << "不满足匹配条件" << endl;
    }


    else if (spinrect_size >= 2)
    {
        double lightbar_angle[contours_filter.size()];
        double spinrect_area[spinrect.size()];
        double spinrect_angle[spinrect.size()];
        double spinrect_length[spinrect.size()];
        double spinrect_width[spinrect.size()];
        double spinrect_ratio[spinrect.size()];


        //轮廓的上下顶点
        vector<Point2f> point_top;
        vector<Point2f> point_bottom;
        vector<Point2f> point_left;
        vector<Point2f> point_right;
        vector<Point2f> point_center;

//获取单个灯条信息
        for (int i = 0; i < spinrect.size(); i++)
        {
            spinrect_area[i] = spinrect[i].size.height*spinrect[i].size.width;
            spinrect_angle[i] = spinrect[i].angle;
            point_center.push_back(spinrect[i].center);
            point_top.push_back(get_topPoint(contours_filter[i]));
            point_bottom.push_back(get_bottomPoint(contours_filter[i]));
            point_left.push_back(get_leftPoint(contours_filter[i]));
            point_right.push_back(get_rightPoint(contours_filter[i]));
            lightbar_angle[i] = fastAtan2(point_top[i].y - point_bottom[i].y, point_top[i].x - point_bottom[i].x);
            spinrect_length[i] = getDistance(point_top[i], point_bottom[i]);
            spinrect_width[i] = getDistance(point_left[i], point_right[i]);
            spinrect_ratio[i] = (spinrect_width[i] / spinrect_length[i]);
        }



        for (int i = 0; i < spinrect_size - 1; i++)
        {
            for (int j = i + 1; j < spinrect_size ;j++)
            {
                //定义灯条（轮廓）角度差
                double lightbar_angle_deviation[spinrect_size][spinrect_size];
                lightbar_angle_deviation[i][j] = fabs(lightbar_angle[i] - lightbar_angle[j]);
                cout<<"角度差(灯条): "<<lightbar_angle_deviation[i][j]<<endl;
                //定义灯条（旋转矩形）角度差
                double angle_deviation[spinrect_size][spinrect_size];
                angle_deviation[i][j] = fabs(spinrect_angle[i] - spinrect_angle[j]);
                cout<<"角度差(旋转矩形): "<<angle_deviation[i][j]<<endl;
                //定义灯条面积比
                double area_ratio[spinrect_size][spinrect_size];
                area_ratio[i][j] = (min(spinrect_area[i],spinrect_area[j]) / max(spinrect_area[i],spinrect_area[j]));
                cout<<"灯条面积比: "<<area_ratio[i][j]<<endl;
                //定义配对灯条中心高度差
                double center_distance_delta[spinrect_size][spinrect_size];
                center_distance_delta[i][j] = fabs(spinrect[i].center.y - spinrect[j].center.y);
                cout<<"灯条中心高度差: "<<center_distance_delta[i][j]<<endl;
                //定义装甲板的长度差和宽度差
                double armor_length_delta[spinrect_size][spinrect_size];
                double armor_width_delta[spinrect_size][spinrect_size];
                armor_length_delta[i][j] = fabs(spinrect_length[i] - spinrect_length[j]);
                armor_width_delta[i][j] = fabs(getDistance(point_top[i], point_top[j]) - getDistance(point_bottom[i], point_bottom[j]));
                cout<<"装甲板长度差: "<<armor_length_delta[i][j]<<endl;
                cout<<"装甲板宽度差: "<<armor_width_delta[i][j]<<endl;

                //定义装甲板长宽比
                double armor_length[spinrect_size][spinrect_size];
                double armor_width[spinrect_size][spinrect_size];
                double armor_ratio[spinrect_size][spinrect_size];
                armor_width[i][j] = ((spinrect_length[i] + spinrect_length[j]) / 2 );
                armor_length[i][j] = ((getDistance(point_top[i], point_top[j]) + getDistance(point_bottom[i], point_bottom[j])) / 2 );
                armor_ratio[i][j] = (armor_width[i][j]/armor_length[i][j]);
                cout<<"装甲板长宽比: "<<armor_ratio[i][j]<<endl;



                //开始二次筛选
                if(center_distance_delta[i][j] > (spinrect_length[i] + spinrect_length[j])/2){continue;}//过滤中心点纵向距离太远的
                if(armor_length_delta[i][j] > armor_length[i][j] / 5 | armor_width_delta[i][j] > armor_width[i][j] / 5 ){continue;}
                if (armor_ratio[i][j] > 0.56 | armor_ratio[i][j] < 0.1){ continue; }
                if(area_ratio[i][j] < 0.2 )
                {
                    if(spinrect_ratio[i] > 0.1 & spinrect_ratio[j] > 0.1){ continue; }
                }
                if(lightbar_angle_deviation[i][j] > 10) { continue; }
                matched_spinrect.push_back(spinrect[i]);
                matched_spinrect.push_back(spinrect[j]);

                }
            }

        }
    return matched_spinrect;
}

