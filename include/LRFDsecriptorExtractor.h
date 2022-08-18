//
// Created by che on 2022/7/29.
//

#ifndef POINTCLOUDREGISTRATION_LRFDSECRIPTOREXTRACTOR_H
#define POINTCLOUDREGISTRATION_LRFDSECRIPTOREXTRACTOR_H
void LRFDescriptorExtractor(pcl::PointCloud<pcl::PointXYZ>::Ptr &inputCloud,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &inputKeyCloud,
                            pcl::search::KdTree<pcl::PointXYZ>::Ptr &kdTree,
                            double searchRadius,
                            Eigen::Matrix<double,32*3,Eigen::Dynamic> &Descriptor);

void computeDescriptorSimilarity(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfSource,
                                 Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfTarget,
                                 double matchThreshold,Eigen::Matrix<double,Eigen::Dynamic,3> &match_result);

void computeTransFromMatch(pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                           Eigen::Matrix<double,Eigen::Dynamic,3> match_result,
                           Eigen::Matrix4d &TranformMatrix);

void computeDescriptorSimilarity_biside(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfSource,
                                        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfTarget,
                                        double matchThreshold,Eigen::Matrix<double,Eigen::Dynamic,3> &match_result);
#endif //POINTCLOUDREGISTRATION_LRFDSECRIPTOREXTRACTOR_H
