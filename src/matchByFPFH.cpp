//
// Created by che on 2022/8/3.
//
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <Eigen/Eigenvalues>
#include <pcl/common/transforms.h>
#include <chrono>

#include "../include/pointTools.h"
#include "../include/pointTransform.h"
#include "../include/matchByFPFH.h"
#include "../include/LRFDsecriptorExtractor.h"

void comput_normal(double radius,pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
        pcl::PointCloud<pcl::Normal>::Ptr &output_normal){

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne;
    ne.setInputCloud(inputCloud);
    ne.setSearchMethod(kdtree);
    ne.setRadiusSearch(radius);
    ne.compute(*output_normal);

}

void comput_descriptor(double radius,pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud,
                       pcl::PointCloud<pcl::Normal>::Ptr inputNoraml,
                       pcl::PointCloud<pcl::FPFHSignature33>::Ptr &Descriptor){

    pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchSurface(inputCloud);
    fpfh.setInputCloud(keyPointCloud);
    fpfh.setInputNormals(inputNoraml);
    fpfh.setRadiusSearch(radius);
    fpfh.setSearchMethod(kdtree);
    fpfh.compute(*Descriptor);
}


void matchByFPFH(double radius,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud1,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud2,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr &matchCloudResult){

    int const descriptorLong = 33;
    //clock_t time_start = clock();

    pcl::PointCloud<pcl::Normal>::Ptr cloudNormal1(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormal2(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr Descriptor1(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr Descriptor2(new pcl::PointCloud<pcl::FPFHSignature33>);

    comput_normal(radius,cloud1,cloudNormal1);
    comput_normal(radius,cloud2,cloudNormal2);

    comput_descriptor(radius,cloud1,keyPointCloud1,cloudNormal1,Descriptor1);
    comput_descriptor(radius,cloud2,keyPointCloud2,cloudNormal2,Descriptor2);


    Eigen::Matrix<double,descriptorLong,Eigen::Dynamic> descriptorMatrix1;
    Eigen::Matrix<double,descriptorLong,Eigen::Dynamic> descriptorMatrix2;
    Eigen::VectorXd descriptorVector = Eigen::VectorXd(descriptorLong);

    int Descriptor1_long = Descriptor1->points.size();
    int Descriptor2_long = Descriptor2->points.size();

    descriptorMatrix1.resize(descriptorLong,Descriptor1_long);
    descriptorMatrix2.resize(descriptorLong,Descriptor2_long);


    for (int i = 0; i < Descriptor1_long; ++i) {

        for (int j = 0; j < descriptorLong; ++j) {

            descriptorVector(j) = Descriptor1->points[i].histogram[j];
        }
        descriptorMatrix1.col(i) = descriptorVector;
    }


    for (int i = 0; i < Descriptor2_long; ++i) {

        for (int j = 0; j < descriptorLong; ++j) {

            descriptorVector(j) = Descriptor2->points[i].histogram[j];
        }
        descriptorMatrix2.col(i) = descriptorVector;
    }

    /*std::cout << "*********FPFH debugging information*********" << std::endl;
    std::cout <<"FPFH Descriptor1 size:(" <<Descriptor1->points.size() << ","
              << Descriptor1->points[0].descriptorSize() <<")" << std::endl;

    std::cout <<"FPFH Descriptor2 size:(" <<Descriptor2->points.size() << ","
              << Descriptor2->points[0].descriptorSize() <<")" << std::endl;

    std::cout <<"FPFH Descriptor2Matrix size:(" << descriptorMatrix2.rows()<< ","
              << descriptorMatrix2.cols() <<")" << std::endl;*/

    Eigen::Matrix<double,Eigen::Dynamic,3> match_result;
    computeDescriptorSimilarity(descriptorMatrix1,descriptorMatrix2,0.8,match_result);

    Eigen::Matrix4d TranformMatrix;
    computeTransFromMatch(keyPointCloud1,keyPointCloud2,match_result,TranformMatrix);

    pcl::transformPointCloud(*cloud1,*matchCloudResult,TranformMatrix);

   /* std::cout << "the time of FPFH process is:" << (clock()-time_start) / (double) CLOCKS_PER_SEC
              << "ms" <<std::endl;*/

    visualization2(cloud2,matchCloudResult,"match_result_by_FPFH");

    //computeTransformError(cloud_clip,matchCloudResult);
    computeTransformErrorByCoodinates(cloud_clip,matchCloudResult);

}
