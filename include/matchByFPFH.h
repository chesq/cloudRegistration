//
// Created by che on 2022/8/3.
//

#ifndef POINTCLOUDREGISTRATION_MATCHBYFPFH_H
#define POINTCLOUDREGISTRATION_MATCHBYFPFH_H
void comput_normal(double radius,pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                   pcl::PointCloud<pcl::Normal>::Ptr &output_normal);

void comput_descriptor(double radius,pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud,
                       pcl::PointCloud<pcl::Normal>::Ptr inputNoraml,
                       pcl::PointCloud<pcl::FPFHSignature33> &Descriptor);

void matchByFPFH(double radius,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud1,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCLoud2,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr originalCloud,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr &matchCloudResult);
#endif //POINTCLOUDREGISTRATION_MATCHBYFPFH_H
