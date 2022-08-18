//
// Created by che on 2022/8/4.
//

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Eigenvalues>
#include <vector>
#include "../include/TOLDI.h"
#include "../include/LRFDsecriptorExtractor.h"
#include "../include/pointTools.h"
#include "../include/pointTransform.h"

void matchByTOLDI(double radius,int bin_mun,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,
                  std::vector<int> indices1,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud1,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
                  std::vector<int> indices2,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud2,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr &matchCloudResult
                 ){

    //clock_t time_start_shot = clock();
    std::vector<std::vector<float>> Histograms1;
    std::vector<std::vector<float>> Histograms2;


    TOLDI_compute(cloud1,indices1,radius,bin_mun,Histograms1);
    TOLDI_compute(cloud2,indices2,radius,bin_mun,Histograms2);


    int cloud1_toldi_number = Histograms1.size();
    int cloud2_toldi_number = Histograms2.size();
    int const descriptor_number = 3*20*20;
    Eigen::VectorXd toldi_point = Eigen::VectorXd(descriptor_number);
    Eigen::Matrix<double,descriptor_number,Eigen::Dynamic> cloud1_desc_Matrix;
    Eigen::Matrix<double,descriptor_number,Eigen::Dynamic> cloud2_desc_Matrix;


    cloud1_desc_Matrix.resize(descriptor_number,cloud1_toldi_number);
    cloud2_desc_Matrix.resize(descriptor_number,cloud2_toldi_number);

    for(int i=0;i<cloud1_toldi_number;i++){

        for(int j=0;j< descriptor_number;j++){
            toldi_point(j) = Histograms1[i][j];
        }
        cloud1_desc_Matrix.col(i) = toldi_point;
    }

    for(int k=0;k<cloud2_toldi_number;k++){

        for(int j=0;j<descriptor_number;j++){
            toldi_point(j) = Histograms2[k][j];
        }
        cloud2_desc_Matrix.col(k) = toldi_point;
    }


   /* std::cout << "***********TOLDI debugging information***********" << std::endl;
    std::cout <<"Histogram1' size is:" << cloud1_desc_Matrix.cols() << std::endl;
    std::cout <<"Histogram2' size is:" << cloud2_desc_Matrix.cols() << std::endl;*/

    Eigen::Matrix<double,Eigen::Dynamic,3> match_result;

    computeDescriptorSimilarity(cloud1_desc_Matrix,cloud2_desc_Matrix,1.3,match_result);

    Eigen::Matrix4d TranformMatrix;
    computeTransFromMatch(keyPointCloud1,keyPointCloud2,match_result,TranformMatrix);

    //pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud1,*matchCloudResult,TranformMatrix);

    /*std::cout << "*********TOLDI match dehugging information*********" << std::endl;
    std::cout << TranformMatrix <<std::endl;*/


    //std::cout << "the time of TOLDI process is:" << (clock()-time_start_shot) / (double) CLOCKS_PER_SEC << "ms" <<std::endl;

    visualization2(cloud2,matchCloudResult,"match_result_by_TOLDI");
    //computeTransformError(cloud_clip,matchCloudResult);
    computeTransformErrorByCoodinates(cloud_clip,matchCloudResult);

}
