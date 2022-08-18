//
// Created by che on 2022/7/27.
//

#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>

#include "../include/clippingAndFilter.h"

void clipCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,pcl::PointCloud<pcl::PointXYZ>::Ptr &clipResult){

    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>);
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("y",pcl::ComparisonOps::GT,-0.15)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("y",pcl::ComparisonOps::LT,1)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("x",pcl::ComparisonOps::GT,-0.15)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("x",pcl::ComparisonOps::LT,1)));
    pcl::ConditionalRemoval<pcl::PointXYZ> condrem(range_cond,false);
    condrem.setInputCloud(inputCloud);
    condrem.setKeepOrganized(false);
    condrem.filter(*clipResult);

}
