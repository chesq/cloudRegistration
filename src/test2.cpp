//
// Created by che on 2022/8/8.
//

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigenvalues>
#include <pcl/io/io.h>
#include <chrono>

#include "../include/pointTools.h"
#include "../include/pointTransform.h"
#include "../include/clippingAndFilter.h"
#include "../include/registrateByICP.h"
#include "../include/LRFDsecriptorExtractor.h"
#include "../include/matchBySHOT.h"
#include "../include/matchByFPFH.h"
#include "../include/TOLDI.h"
#include "../include/3DHoPD.h"

int main(){

    //std::string file_path1 = "../data/pointcloud/frame-000.depth.png";
    std::string file_path1 = "../data/pointcloud/kitti_280.png";
    std:string file_path_ply = "../data/pointcloud/cloud2.ply";
    std::string file_RGB_path1 = "../data/pointcloud/frame-000.color.png";
    std::string file_path2 = "../data/pointcloud/frame-001.depth.png";
    std::string file_RGB_path2 = "../data/pointcloud/frame-001.color.png";
    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudResult(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud1_down(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud2_down(new  pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sourceCloud1_rgb(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sourceCloud2_rgb(new pcl::PointCloud<pcl::PointXYZRGBA>);

    //createCloudFromDepthImage(file_path1,file_RGB_path1,sourceCloud1,sourceCloud1_rgb);
    //createCloudFromDepthImage(file_path2,file_RGB_path2,sourceCloud2,sourceCloud2_rgb);
    createCloudFromPly(file_path_ply,sourceCloud1);
    Eigen::Matrix3d rotationMatrix;
    Eigen::Vector3d translationVector (0,0,0);//set the RT matrix for test
    //Eigen::Quaterniond q1(1,0.1,0.4,0.7);  //set the RT matrix for test
    Eigen::Quaterniond q1(1,0.02,0.002,0.002);
    q1.normalize();
    rotationMatrix = q1.toRotationMatrix();

    cloudResult = cloudTransform(sourceCloud1,rotationMatrix,translationVector);

    visualization2(sourceCloud1,cloudResult,"original cloud");
    visualization2(sourceCloud1,sourceCloud1,"ground_truth");

    //visualization1_rgbd(sourceCloud1_rgb);

    cloudDownSample(sourceCloud1,0.03,sourceCloud1_down);
    cloudDownSample(cloudResult,0.03,sourceCloud2_down);
    std::cout<<"down sample size:" << sourceCloud1_down->size()<<std::endl;
    std::cout<<"down sample size:" << sourceCloud2_down->size()<<std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointsCloud(new  pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloudOfTarget(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree1(new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<int> indices1;
    std::vector<int> indices2;

    clock_t time_start_our = clock(); //compute the time cost of OURS process.

    iss_detector(sourceCloud1_down,keyPointsCloud,indices1,kdTree1);
    iss_detector(sourceCloud2_down,keyPointCloudOfTarget,indices2,kdTree1);
    //visualization2(sourceCloud1_down,keyPointsCloud,"keypoints cloud1");
    //visualization2(sourceCloud2_down,keyPointCloudOfTarget,"keypoints cloud2");

    std::cout << "the time of iss is:" << (clock()-time_start_our)/(double) CLOCKS_PER_SEC << "ms" <<std::endl;
    Eigen::Matrix<double,32*3,Eigen::Dynamic> DescriptorOfSource;
    Eigen::Matrix<double,32*3,Eigen::Dynamic> DescriptorOfTarget;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree2(new pcl::search::KdTree<pcl::PointXYZ>);
    LRFDescriptorExtractor(sourceCloud1_down,keyPointsCloud,kdTree2,0.6,DescriptorOfSource);//0.4
    LRFDescriptorExtractor(sourceCloud2_down,keyPointCloudOfTarget,kdTree2,0.6,DescriptorOfTarget);//0.4

    std::cout << "descriptor extrctor is done" << std::endl;
    Eigen::Matrix<double,Eigen::Dynamic,3> match_result;
    computeDescriptorSimilarity(DescriptorOfTarget,DescriptorOfSource,0.05,match_result);//0.05
    Eigen::Matrix4d TranformMatrix;
    computeTransFromMatch(keyPointCloudOfTarget,keyPointsCloud,match_result,TranformMatrix);

    pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::transformPointCloud(*cloudResult,*matchCloudResult,TranformMatrix);

    computeTransformErrorByCoodinates(sourceCloud1,matchCloudResult);
    std::cout << "the time of our process is:" << (clock()-time_start_our)/(double) CLOCKS_PER_SEC << "ms" <<std::endl;

    visualization_correspondence(sourceCloud1_down,keyPointsCloud,sourceCloud2_down,keyPointCloudOfTarget,match_result);
    visualization2(sourceCloud1,matchCloudResult,"match_result_By_Our");

    clock_t time_start_shot = clock();
    //0.4
    matchBySHOT(0.4,sourceCloud2_down,sourceCloud1_down,keyPointCloudOfTarget,keyPointsCloud,sourceCloud2_down,matchCloudResult);
    std::cout << "the time of shot process is:" << (clock()-time_start_shot)/(double) CLOCKS_PER_SEC << "ms" <<std::endl;
    computeTransformErrorByCoodinates(sourceCloud1,matchCloudResult);
    visualization2(sourceCloud1,matchCloudResult,"match_result_By_shot");


    clock_t time_start_fpfh= clock();
    //0.3
    matchByFPFH(0.4,sourceCloud2_down,sourceCloud1_down,keyPointCloudOfTarget,keyPointsCloud,sourceCloud1_down,matchCloudResult);
    std::cout << "the time of fpfh process is:" << (clock()-time_start_fpfh)/(double) CLOCKS_PER_SEC << "ms" <<std::endl;
    visualization2(sourceCloud1,matchCloudResult,"match_result_By_fpfh");

    clock_t time_start_toldi= clock();
    //2
    matchByTOLDI(3,20,sourceCloud2_down,indices2,keyPointCloudOfTarget,
                 sourceCloud1_down,indices1,keyPointsCloud,sourceCloud2_down,matchCloudResult);
    std::cout << "the time of toldi process is:" << (clock()-time_start_toldi)/(double) CLOCKS_PER_SEC << "ms" <<std::endl;
    visualization2(sourceCloud1,matchCloudResult,"match_result_By_TOLDI");

    clock_t time_start_3dhpod = clock();
    matchBy3DHoPD(sourceCloud2_down,sourceCloud1_down,sourceCloud2_down,cloudResult,matchCloudResult);
    std::cout << "the time of 3dhopd process is:" << (clock()-time_start_3dhpod)/(double) CLOCKS_PER_SEC << "ms" <<std::endl;
    visualization2(sourceCloud1,matchCloudResult,"match_result_By_3dhopd");

    return 0;
}

