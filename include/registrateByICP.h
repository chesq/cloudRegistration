//
// Created by che on 2022/7/28.
//

#ifndef POINTCLOUDREGISTRATION_REGISTRATEBYICP_H
#define POINTCLOUDREGISTRATION_REGISTRATEBYICP_H
#include <pcl/registration/icp.h>
#include <pcl/point_types.h>
#include <Eigen/Eigenvalues>

void findCorrespondencesPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &target,pcl::PointCloud<pcl::PointXYZ>::Ptr &outSource,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &outTarget);

Eigen::Affine3d computeTransformMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
                                       pcl::PointCloud<pcl::PointXYZ>::Ptr &target);

pcl::PointCloud<pcl::PointXYZ>::Ptr icpCloudRegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr &sourceCLoud,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr &targetCloud,
                          int iterateNumber);

#endif //POINTCLOUDREGISTRATION_REGISTRATEBYICP_H
