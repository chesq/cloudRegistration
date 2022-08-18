//
// Created by che on 2022/7/27.
//

#ifndef POINTCLOUDREGISTRATION_POINTTOOLS_H
#define POINTCLOUDREGISTRATION_POINTTOOLS_H
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>



void createCloudFromTxt(const std::string file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

void writeCloudToTxt(pcl::PointCloud<pcl::PointXYZ>:: Ptr &cloud, std::string file_write_path);

void visualization1(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void visualization2(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,std::string viewName);

void visualization_correspondence(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyCloud1,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyCloud2,
                                  Eigen::Matrix<double,Eigen::Dynamic,3> match_result);

void iss_detector(pcl::PointCloud<pcl::PointXYZ>::Ptr &inputCloud,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr &keyPoints,
                  std::vector<int> &indices,
                  pcl::search::KdTree<pcl::PointXYZ>::Ptr inputKdTree);

void cloudAddNoisy(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,double mu,double sigma,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr &outCloud);

void cloudDownSample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,float size,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &outCloud);

void compute_precision_recall(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr clip_cloud,
                              std::vector<float> threshold,std::string method);

void getFileName(std::string file_path,std::vector<std::string> &file_name);

void writeVectorToTxt(Eigen::VectorXd our_dis_vector, std::string file_write_path);

void createCloudFromPly(const std::string file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

void createCloudFromDepthImage(std::string depth_file_path,std::string img_file_path,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBD);


void visualization1_rgbd(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr inputcloud);

void visualization2_rgbd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2,std::string viewName);


#endif //POINTCLOUDREGISTRATION_POINTTOOLS_H
