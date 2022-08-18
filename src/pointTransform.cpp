//
// Created by che on 2022/7/27.
//

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigenvalues>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <vector>
#include <math.h>
#include "../include/pointTransform.h"

pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTransform(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                   Eigen::Matrix3d rotationMatrix,
                                                   Eigen::Vector3d translationVector){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudResult(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Affine3d transform_2 = Eigen::Affine3d::Identity();
    transform_2.translation()<<translationVector.x(),translationVector.y(),translationVector.z();

    transform_2.rotate(rotationMatrix);

    //std::cout<<"rotation matrix is:"<< "\n" <<transform_2.matrix()<<std::endl;
    pcl::transformPointCloud(*cloud,*cloudResult,transform_2);

    return cloudResult;

}


void computeTransformError(pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &computeTarget){

    int size_n = target->size();
    Eigen::Matrix<double,3,Eigen::Dynamic> matrixP(3,size_n),matrixQ(3,size_n);
    Eigen::Vector3d eulerAngle,translation;
    Eigen::Affine3d TransformMatrix;
    double xError,yError,zError,transError;
    const double pi = 3.1415926;


    for (int j = 0; j < size_n; ++j) {
        matrixP(0,j) = target->points[j].x;
        matrixP(1,j) = target->points[j].y;
        matrixP(2,j) = target->points[j].z;

        matrixQ(0,j) = computeTarget->points[j].x;
        matrixQ(1,j) = computeTarget->points[j].y;
        matrixQ(2,j) = computeTarget->points[j].z;

    }

    TransformMatrix = Eigen::umeyama(matrixP,matrixQ, false);
    eulerAngle = TransformMatrix.rotation().eulerAngles(2,1,0);
    translation = TransformMatrix.translation();

    zError = eulerAngle[0] *180 / pi;
    yError = eulerAngle[1] *180 / pi;
    xError = eulerAngle[2] *180 / pi;
    transError = translation.norm();

    //std::cout<<"error matrix is:" << "\n" << TransformMatrix.matrix() << std::endl;
    //std::cout<< "x,y,z direction angle error are:" << "\n" << xError<<","<< yError <<","<< zError << std::endl;
    //std::cout<< "translation error is:" << "\n" << transError << std::endl;

}

double computeTransformErrorByCoodinates(pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr computeTarget){

    size_t n = target->size();
    double rmse = 0;
    Eigen::Vector3d ErrorVector;
    ErrorVector[0] = 0;
    ErrorVector[1] = 0;
    ErrorVector[2] = 0;

    for(size_t j = 0;j < n; ++j) {
        ErrorVector[0] += (target->points[j].x-computeTarget->points[j].x) * (target->points[j].x-computeTarget->points[j].x);
        ErrorVector[1] += (target->points[j].y-computeTarget->points[j].y) * (target->points[j].y-computeTarget->points[j].y);
        ErrorVector[2] += (target->points[j].z-computeTarget->points[j].z) * (target->points[j].z-computeTarget->points[j].z);

    }

    std::cout<<"error of x axis:" << ErrorVector[0]<<std::endl;
    std::cout<<"error of y axis:" << ErrorVector[1]<<std::endl;
    std::cout<<"error of z axis:" << ErrorVector[2]<<std::endl;

    rmse =  std::sqrt((ErrorVector[0] + ErrorVector[1] + ErrorVector[2]) / n);
    std::cout<<"rmse:" << rmse <<std::endl;

    return rmse;

}

