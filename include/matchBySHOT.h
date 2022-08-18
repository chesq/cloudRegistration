//
// Created by che on 2022/8/2.
//

#ifndef POINTCLOUDREGISTRATION_MATCHBYSHOT_H
#define POINTCLOUDREGISTRATION_MATCHBYSHOT_H
void compute_normals(float radius, pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                     pcl::PointCloud<pcl::Normal>::Ptr &cloud1_normals,
                     pcl::PointCloud<pcl::Normal>::Ptr &cloud2_normals);

void calculate_SHOT(float radius,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_keypoints,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_keypoints,
                    pcl::PointCloud<pcl::Normal>::Ptr cloud1_normals,
                    pcl::PointCloud<pcl::Normal>::Ptr cloud2_normals,
                    pcl::PointCloud<pcl::SHOT352>::Ptr &cloud1_shot,
                    pcl::PointCloud<pcl::SHOT352>::Ptr &cloud2_shot
);

void matchBySHOT(float radius,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_keypoints,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_keypoints,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr &matchCloudResult
);
#endif //POINTCLOUDREGISTRATION_MATCHBYSHOT_H
