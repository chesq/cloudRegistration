//
// Created by che on 2022/7/28.
//

#include <pcl/registration/icp.h>
#include <pcl/point_types.h>
#include <Eigen/Eigenvalues>
#include "../include/registrateByICP.h"
#include "../include/pointTools.h"

void findCorrespondencesPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &target,pcl::PointCloud<pcl::PointXYZ>::Ptr &outSource,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &outTarget){

    pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
    kdTree.setInputCloud(target);
    std::vector<int> index;
    std::vector<float> dist;

    for(int i = 0;i < source->size();i++){
        kdTree.nearestKSearch(source->points[i],2,index,dist);
        if (dist[0]<0.1){
            outTarget->push_back(target->points[index[0]]);
            outSource->push_back(source->points[i]);

        }
    }

    //std::cout<<"total correspondence points are:" << outTarget->points.size()<<std::endl;
}


Eigen::Affine3d computeTransformMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &target){

    int size_n = source->size();
    Eigen::Affine3d TransformMatrix;
    Eigen::Matrix<double,3,Eigen::Dynamic> matrixP(3,size_n),matrixQ(3,size_n);

    for (int j = 0; j < size_n; ++j) {
        matrixP(0,j) = source->points[j].x;
        matrixP(1,j) = source->points[j].y;
        matrixP(2,j) = source->points[j].z;

        matrixQ(0,j) = target->points[j].x;
        matrixQ(1,j) = target->points[j].y;
        matrixQ(2,j) = target->points[j].z;

    }

    TransformMatrix = Eigen::umeyama(matrixP,matrixQ, false); //DO NOT miss the param with_scaling!!!

    return TransformMatrix;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr icpCloudRegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr &sourceCLoud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &targetCloud,
        int iterateNumber){

    pcl::PointCloud<pcl::PointXYZ>::Ptr outSource(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr outTarget(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_tran;
    source_tran = sourceCLoud;
    Eigen::Affine3d transformMatrix = Eigen::Affine3d::Identity();

    //visualization2(source_tran,targetCloud);

    for(int i = 0;i < iterateNumber;i++){
        outSource->clear();
        outTarget->clear();
        findCorrespondencesPoint(source_tran,targetCloud,outSource,outTarget);
        transformMatrix = computeTransformMatrix(outSource,outTarget);

        pcl::transformPointCloud(*source_tran,*source_tran,transformMatrix);

    }
    std::cout << "transfromMatrix is:" << "\n" << transformMatrix.matrix() << std::endl;


    //visualization2(source_tran,targetCloud);
    return source_tran;

}
