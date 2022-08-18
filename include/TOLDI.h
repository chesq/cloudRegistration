//
// Created by che on 2022/8/4.
//

#ifndef POINTCLOUDREGISTRATION_TOLDI_H
#define POINTCLOUDREGISTRATION_TOLDI_H

//#define Pi 3.1415926
#define NULL_POINTID -1
#define TOLDI_NULL_PIXEL 100
//
using namespace std;
#include <pcl/point_types.h>
typedef struct{
    float x;
    float y;
    float z;
}Vertex;
typedef struct{
    int pointID;
    Vertex x_axis;
    Vertex y_axis;
    Vertex z_axis;
}LRF;
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
//TOLDI_LRF
void TOLDI_LRF_Z_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,Vertex &z_axis);
void TOLDI_LRF_X_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,Vertex z_axis,float sup_radius,vector<float> PointDist,Vertex &x_axis);
void TOLDI_LRF_Y_axis(Vertex x_axis,Vertex z_axis,Vertex &y_axis);
void TOLDI_LRF_for_cloud_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,vector<int> indices,float sup_radius,vector<LRF>&Cloud_LRF);
//TOLDI_descriptor
void transformCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,LRF pointLRF,pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed_cloud);
void local_TOLDI(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,float sup_radius,int bin,vector<float>&histogram);
void TOLDI_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,vector<int>indices,float sup_radius,int bin_num,vector<vector<float>>&Histograms);

void matchByTOLDI(double radius,int bin_mun,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,
                  std::vector<int> indices1,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud1,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
                  std::vector<int> indices2,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud2,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr &matchCloudResult
);
#endif
