//
// Created by che on 2022/7/27.
//

#ifndef POINTCLOUDREGISTRATION_POINTTRANSFORM_H
#define POINTCLOUDREGISTRATION_POINTTRANSFORM_H

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTransform(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                   Eigen::Matrix3d rotationMatrix,Eigen::Vector3d translationVector);

void computeTransformError(pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr &computeTarget);

double computeTransformErrorByCoodinates(pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr computeTarget);

#endif //POINTCLOUDREGISTRATION_POINTTRANSFORM_H
