//
// Created by che on 2022/8/2.
//
#include <pcl/point_types.h>
#include <Eigen/Eigenvalues>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <vector>
#include <chrono>

#include "../include/matchBySHOT.h"
#include "../include/LRFDsecriptorExtractor.h"
#include "../include/pointTools.h"
#include "../include/pointTransform.h"

// compute the normal

void compute_normals(float radius,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                     pcl::PointCloud<pcl::Normal>::Ptr &cloud1_normals,
                     pcl::PointCloud<pcl::Normal>::Ptr &cloud2_normals)
{

    // Estimate the normals.
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setRadiusSearch(radius);
    normalEstimation.setNumberOfThreads(1);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);

    normalEstimation.setInputCloud(sourceCloud);
    normalEstimation.compute(*cloud1_normals);

    normalEstimation.setInputCloud(targetCloud);
    normalEstimation.compute(*cloud2_normals);
}

void calculate_SHOT(float radius,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_keypoints,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_keypoints,
                    pcl::PointCloud<pcl::Normal>::Ptr cloud1_normals,
                    pcl::PointCloud<pcl::Normal>::Ptr cloud2_normals,
                    pcl::PointCloud<pcl::SHOT352>::Ptr &cloud1_shot,
                    pcl::PointCloud<pcl::SHOT352>::Ptr &cloud2_shot
                    )
{

    // SHOT estimation object.
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    shot.setRadiusSearch(radius);
    shot.setNumberOfThreads(1);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    shot.setSearchMethod(kdtree);

    shot.setInputCloud(cloud1_keypoints);
    shot.setSearchSurface(sourceCloud);
    shot.setInputNormals(cloud1_normals);
    shot.compute(*cloud1_shot);

    shot.setInputCloud(cloud2_keypoints);
    shot.setSearchSurface(targetCloud);
    shot.setInputNormals(cloud2_normals);
    shot.compute(*cloud2_shot);
}


void matchBySHOT(float radius,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_keypoints,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_keypoints,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr &matchCloudResult
                 )
{
    clock_t time_start_shot = clock();
    pcl::PointCloud<pcl::Normal>::Ptr cloud1_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud2_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::SHOT352>::Ptr cloud1_shot(new pcl::PointCloud<pcl::SHOT352>);
    pcl::PointCloud<pcl::SHOT352>::Ptr cloud2_shot(new pcl::PointCloud<pcl::SHOT352>);
    pcl::SHOT352 shot_point;
    Eigen::VectorXd shot_vector = Eigen::VectorXd(352);
    Eigen::Matrix<double,352,Eigen::Dynamic>shotMatrixSource,shotMatrixTarget;


    compute_normals(radius,sourceCloud,targetCloud,cloud1_normals,cloud2_normals);

    calculate_SHOT(radius,sourceCloud,targetCloud,cloud1_keypoints,cloud2_keypoints,
                   cloud1_normals,cloud2_normals,cloud1_shot,cloud2_shot);

    int cloud1_shot_number = cloud1_shot->size();
    int cloud2_shot_number = cloud2_shot->size();

    shotMatrixSource.resize(352,cloud1_shot_number);
    shotMatrixTarget.resize(352,cloud2_shot_number);

    for(int i=0;i<cloud1_shot_number;i++){

        shot_point = cloud1_shot->points[i];
        for(int j=0;j<352;j++){
            shot_vector(j) = shot_point.descriptor[j];
        }
        shotMatrixSource.col(i) = shot_vector;

    }

    for(int k=0;k<cloud2_shot_number;k++){

        shot_point = cloud2_shot->points[k];
        for(int j=0;j<352;j++){
            shot_vector(j) = shot_point.descriptor[j];
        }
        shotMatrixTarget.col(k) = shot_vector;
    }

    //Eigen::VectorXd cloud1_shot_vector = Eigen::VectorXd(352);  //Do not work;
    //cloud1_shot_vector = cloud1_shot->points[0].descriptor;  //Do not work
    std::cout << "************shot debugging information************" << std::endl;
    std::cout << "source cloud SHOT descriptor size is: (" << shotMatrixSource.rows()<<","
    <<shotMatrixSource.cols() << ")" << std::endl;
    std::cout << "target cloud SHOT descriptor size is: (" << shotMatrixTarget.rows()<<","
              <<shotMatrixTarget.cols() << ")" << std::endl;

    Eigen::Matrix<double,Eigen::Dynamic,3> match_result;

    computeDescriptorSimilarity(shotMatrixSource,shotMatrixTarget,0.3,match_result);

    Eigen::Matrix4d TranformMatrix;
    computeTransFromMatch(cloud1_keypoints,cloud2_keypoints,match_result,TranformMatrix);

    pcl::transformPointCloud(*sourceCloud,*matchCloudResult,TranformMatrix);
    //std::cout << "*********SHOT match dehugging information*********" << std::endl;
    //std::cout << TranformMatrix <<std::endl;
    //std::cout << "the time of shot process is:" << (clock()-time_start_shot) / (double) CLOCKS_PER_SEC << "ms" <<std::endl;

    visualization2(targetCloud,matchCloudResult,"match_result_by_SHOT");

    //computeTransformError(cloud_clip,matchCloudResult);
    //computeTransformErrorByCoodinates(cloud_clip,matchCloudResult);

}


