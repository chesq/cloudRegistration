//
// Created by che on 2022/7/27.
//

#ifndef POINTCLOUDREGISTRATION_CLIPPINGANDFILTER_H
#define POINTCLOUDREGISTRATION_CLIPPINGANDFILTER_H

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

void clipCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &clipResult);


#endif //POINTCLOUDREGISTRATION_CLIPPINGANDFILTER_H
