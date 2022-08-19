//
// Created by che on 2022/7/27.
//

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <Eigen/Eigenvalues>
#include <boost/thread.hpp>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/filters/voxel_grid.h>
#include <dirent.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../include/pointTools.h"


void createCloudFromTxt(const std::string file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    std::ifstream file(file_path.c_str());
    std::string line;
    pcl::PointXYZ point;
    pcl::Normal normal;

    while (getline(file,line)){
        std::string::iterator it;
        for(it=line.begin();it<line.end();it++){
            if(*it == ','){
                line.erase(it);
                line.insert(it,' '); //insert space;
                it--;
            }

        }
        std::stringstream ss(line);
        ss >> point.x;
        ss >> point.y;
        ss >> point.z;
        ss >> normal.normal_x;
        ss >> normal.normal_y;
        ss >> normal.normal_z;
        cloud->push_back(point);
    }
    file.close();
}

void writeCloudToTxt(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::string file_write_path){
    std::ofstream outfile;
    outfile.open(file_write_path);

    outfile.precision(6);

    if(!outfile.is_open())
        std::cout << "write point cloud file is failed!" << std::endl;


    int size_n = cloud->size();
    if(cloud->size() > 10000)
        size_n = 10000;
    for(int i = 0;i<size_n;i++) {
        outfile << cloud->points[i].x << "," << cloud->points[i].y<< "," <<  cloud->points[i].z << "\n";

    }
    std::cout<< "The pointcloud file was write done" << std::endl;

    outfile.close();

}

void visualization1(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("viewer"));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            single_color(cloud, 136, 125, 110);
    viewer->setBackgroundColor(0,0,0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "testCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             3, "testCloud");
    while (!viewer->wasStopped()) {

        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

void visualization2(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,std::string viewName) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer(viewName));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            single_color1(cloud1, 0, 206, 209);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            single_color2(cloud2,255,200,0);

    viewer->setBackgroundColor(255,255,255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud1, single_color1, "testCloud");
    viewer->addPointCloud<pcl::PointXYZ>(cloud2,single_color2,"testCloud2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             4, "testCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             7, "testCloud2");
    while (!viewer->wasStopped()) {

        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}


void visualization_correspondence(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyCloud1,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyCloud2,
                                  Eigen::Matrix<double,Eigen::Dynamic,3> match_result) {

    pcl::Correspondence corr;
    pcl::Correspondences corrs;

    int num = match_result.rows();

    for (int j = 0; j < num; ++j) {
        corr.index_query = match_result(j,0);
        corr.index_match = match_result(j,1);
        corrs.push_back(corr);

    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("keypoints_match_viewer"));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            single_color1(cloud1, 0, 206, 209);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            single_color2(cloud2,255,200,0);

    viewer->addPointCloud<pcl::PointXYZ>(cloud1, single_color1, "Cloud1");
    viewer->addPointCloud<pcl::PointXYZ>(cloud2, single_color2, "Cloud2");

    viewer->setBackgroundColor(255,255,255);
    viewer->addCorrespondences<pcl::PointXYZ>(keyCloud2,keyCloud1,corrs,"correspondence");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             3, "Cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             3, "Cloud2");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                             5, "correspondence");


    while (!viewer->wasStopped()) {

        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}


void iss_detector(pcl::PointCloud<pcl::PointXYZ>::Ptr &inputCloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &keyPoints,
        std::vector<int> &indices,
        pcl::search::KdTree<pcl::PointXYZ>::Ptr inputKdTree){

    //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdTree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    //pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>());

    pcl::ISSKeypoint3D<pcl::PointXYZ,pcl::PointXYZ> iss_3d;
    pcl::PointIndicesConstPtr pointIndices;
    iss_3d.setInputCloud(inputCloud);
    iss_3d.setSearchMethod(inputKdTree);
    iss_3d.setSalientRadius(0.05);//0.2
    iss_3d.setNonMaxRadius(0.05);//0.2
    iss_3d.setMinNeighbors(10);
    iss_3d.setThreshold21(0.95);    iss_3d.setThreshold32(0.45);
    iss_3d.compute(*keyPoints);
    pointIndices = iss_3d.getKeypointsIndices();
    indices = pointIndices->indices;

    std::cout<< "total keypoints are:" << keyPoints->size() << std::endl;

}

void cloudAddNoisy(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,double mu,double sigma,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr &outCloud){

    boost::mt19937 rng;
    rng.seed(static_cast<unsigned int>(time(0)));
    boost::normal_distribution<>nd(mu,sigma);
    boost::variate_generator<boost::mt19937&,boost::normal_distribution<>> var_nor(rng,nd);

    for (int j = 0; j < inputCloud->size(); ++j) {
        outCloud->points[j].x = inputCloud->points[j].x + static_cast<float>(var_nor());
    }
}

void cloudDownSample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,float size,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &outCloud){

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (inputCloud);
    sor.setLeafSize (size, size, size);
    sor.filter (*outCloud);

}


void compute_precision_recall(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr clip_cloud,
                              std::vector<float> threshold,std::string method){


    //clip_cloud代表ground truth
    pcl::search::KdTree<pcl::PointXYZ> kdtree1;
    pcl::search::KdTree<pcl::PointXYZ> kdtree2;
    std::vector<int> search_index(1);
    std::vector<float> search_dis(1);
    int thre_num = threshold.size();
    int precision_num = 0;
    int recall_num = 0;
    float precision,recall;
    Eigen::Matrix<float,6,2> precision_recall;
    std::vector<float> precisionVector;
    std::vector<float> recallVector;

    kdtree1.setInputCloud(clip_cloud);

    kdtree2.setInputCloud(inputcloud);

    for (int i = 0; i < thre_num; ++i) {
        recall_num = 0;
        precision_num = 0;
        for(pcl::PointCloud<pcl::PointXYZ>::iterator it=inputcloud->begin();it < inputcloud->end();it++)
        {
            kdtree1.nearestKSearch(*it,1,search_index,search_dis);
            if(std::sqrt(search_dis[0]) < threshold[i])
                recall_num +=1;
        }

        for(pcl::PointCloud<pcl::PointXYZ>::iterator it1=clip_cloud->begin();it1 < clip_cloud->end();it1++)
        {
            kdtree2.nearestKSearch(*it1,1,search_index,search_dis);
            if(std::sqrt(search_dis[0]) < threshold[i])
                precision_num +=1;
        }

        precision = precision_num / clip_cloud->size() * 100;
        precision_recall.row(i)(0) = precision;
        recall = recall_num / inputcloud->size() *100;
        precision_recall.row(i)(1) = recall;
    }


    std::cout << "************precision-recall information************" <<std::endl;
    std::cout << method <<":" << std::endl;
    std::cout << precision_recall <<std::endl;

}


void getFileName(std::string file_path,std::vector<std::string> &file_name){

    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if((dir=opendir(file_path.c_str()))==NULL){
        perror("Open dir error...");
        exit(1);
    }

    while((ptr=readdir(dir))!=NULL){
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
            continue;
        else if(ptr->d_type == 8)
            file_name.push_back(ptr->d_name);
        else if (ptr->d_type == 10)
            continue;
        else if(ptr->d_type == 4)
            file_name.push_back(ptr->d_name);

    }
    closedir(dir);

}


void writeVectorToTxt(Eigen::VectorXd our_dis_vector, std::string file_write_path){
    std::ofstream outfile;
    outfile.open(file_write_path);

    outfile.precision(6);

    if(!outfile.is_open())
        std::cout << "write point cloud file is failed!" << std::endl;

    for(int i = 0;i<our_dis_vector.size();i++) {
        outfile << our_dis_vector[i] << "\n";

    }
    std::cout<< "The rmse value was write done" << std::endl;

    outfile.close();

}

void createCloudFromPly(const std::string file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){

    if (pcl::io::loadPLYFile<pcl::PointXYZ>(file_path, *cloud) == -1) {
        //PCL_ERROR("Couldnot read file.\n");
        cout << "Could NOT read file." << endl;
        system("pause");
    }

    cout << "point size is：" << cloud->points.size() << endl;

}

void createCloudFromDepthImage(std::string depth_file_path,std::string img_file_path,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBD){

    cv::Mat depth,img;
    depth = cv::imread(depth_file_path,-1);
    img =  cv::imread(img_file_path);
    pcl::PointXYZ point;
    pcl::PointXYZRGBA pointXyzrgb;
    ushort d;

    //the camera intrinsics
    float camera_factor =  1000;
    float camera_cx = 596.5593;//320.0;//
    float camera_cy = 149.854;//240.0;//
    float camera_fx = 721.53;//533.069;//
    float camera_fy = 721.5377;//533.069;//

    for (int i = 0; i < depth.rows; ++i) {
        for (int j = 0; j <depth.cols ; ++j) {
            d = depth.ptr<ushort>(i)[j];
            if(d == 0 )
                continue;

            point.z = double(d) / camera_factor;
            point.y = (i - camera_cy) * point.z / camera_fy;
            point.x = (j - camera_cx) * point.z / camera_fx;

            pointXyzrgb.x = (j - camera_cx) * point.z / camera_fx;
            pointXyzrgb.y = (i - camera_cy) * point.z / camera_fy;
            pointXyzrgb.z = double(d) / camera_factor;

            pointXyzrgb.r = img.ptr<uchar>(i)[j*3];
            pointXyzrgb.g = img.ptr<uchar>(i)[j*3+1];
            pointXyzrgb.b = img.ptr<uchar>(i)[j*3+2];


            cloud->points.push_back(point);
            cloudRGBD->points.push_back(pointXyzrgb);
        }
    }

    std::cout << cloudRGBD->points[0] << std::endl;

}


void visualization1_rgbd(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr inputcloud){

    pcl::visualization::CloudViewer viewer("viewer");

    viewer.showCloud(inputcloud);
    while (!viewer.wasStopped ()){

    }
}


void visualization2_rgbd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1,pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2,std::string viewName) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer(viewName));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
            single_color1(cloud1, 0, 206, 209);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
            single_color2(cloud2,255,200,0);

    viewer->addPointCloud<pcl::PointXYZRGB>(cloud1, single_color1, "testCloud");
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud2,single_color2,"testCloud2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             4, "testCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             7, "testCloud2");
    while (!viewer->wasStopped()) {

        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}


