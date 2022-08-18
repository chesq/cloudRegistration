//
// Created by che on 2022/7/24. e-mail:chesq_njtu@163.com
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

int main(int argc,char** argv){

    std::string file_path = "../data/pointcloud/person_0001.txt";
    //std::string file_path = "../data/testdata_3dmatch/sun_3d_mit_lab/cloud_bin_0.ply";
    std::string write_path = "../data/pointresult/result1.txt";
    std::string file_clip_path = "../data/pointresult/airplane_clip.txt";
    std::string file_tran_write_path = "../data/pointresult/airplane_tran.txt";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip(new pcl::PointCloud<pcl::PointXYZ>);
    createCloudFromTxt(file_path,cloud);
    //createCloudFromPly(file_path,cloud);
    writeCloudToTxt(cloud_clip,file_tran_write_path);

    //createCloudFromTxt(file_mod_path,cloud_mod);
    //visualization(cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudResult;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudResult_icp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_tran;
    Eigen::Matrix3d rotationMatrix;
    //rotationMatrix << 1,0,0,0.3,0.2,0.1,0,0,1;
    Eigen::Vector3d translationVector (0.01,0.01,0.02);//set the RT matrix for test
    //Eigen::Quaterniond q1(1,0.1,0.4,0.7);  //set the RT matrix for test
    Eigen::Quaterniond q1(1,0.1,0.7,0.4);
    q1.normalize();
    rotationMatrix = q1.toRotationMatrix();
    clipCloud(cloud,cloud_clip);

    writeCloudToTxt(cloud_clip,file_clip_path);
    std::cout<<"originalCloud size is " << cloud->size() << std::endl;
    std::cout<<"clipCloud size is " << cloud_clip->size() << std::endl;

    cloudResult = cloudTransform(cloud_clip,rotationMatrix,translationVector);
    cloudAddNoisy (cloud,0,0.00002,cloud);
    cloudAddNoisy (cloudResult,0,0.00002,cloudResult);
    //cloudDownSample(cloud,0.001,cloud);
    //cloudDownSample(cloudResult,0.001,cloudResult);

    //std::cout<<"DownSample Cloud size is " << cloud->size() << std::endl;
    //std::cout<<"DownSample clipCloud size is " << cloudResult->size() << std::endl;

    writeCloudToTxt(cloudResult,write_path);

    visualization2(cloud,cloudResult,"originalCloud");
    visualization2(cloud,cloud_clip,"ground truth");
    clock_t time_start = clock(); //compute the time cost of icp process.

    cloudResult_icp = cloudTransform(cloud_clip,rotationMatrix,translationVector);
    source_tran = icpCloudRegistration(cloudResult_icp,cloud,30);
    //实验发现这个函数会引用传入的cloudResult_icp，并更改数据，故在此传入一个cloudResult_icp，而不是下面函数要用的cloudResult

    std::cout << "the time of icp process is:" << (clock()-time_start)/(double) CLOCKS_PER_SEC << "ms" <<std::endl;
    //writeCloudToTxt(source_tran,file_tran_write_path);
    //writeCloudToTxt(cloudResult,write_path);

    //computeTransformError(cloud_clip,source_tran);
    //computeTransformErrorByCoodinates(cloud_clip,source_tran);

    visualization2(cloud,source_tran,"icpResult");


    //***********iss3d detector

    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointsCloud(new  pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloudOfTarget(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree1(new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<int> indices1;
    std::vector<int> indices2;

    clock_t iss_start = clock();
    iss_detector(cloudResult,keyPointCloudOfTarget,indices2,kdTree1);
    iss_detector(cloud,keyPointsCloud,indices1,kdTree1);
    std::cout << "the time of iss is:" << (clock()-iss_start)/(double) CLOCKS_PER_SEC << "s" <<std::endl;
    //local reference frame-based descriptor extraction of the keypoints
    //compute the normal and the descriptor
    Eigen::Matrix<double,32*3,Eigen::Dynamic> DescriptorOfSource;
    Eigen::Matrix<double,32*3,Eigen::Dynamic> DescriptorOfTarget;

    clock_t time_start_our = clock(); //compute the time cost of OURS process.
    std::cout <<" computing LRF..." << std::endl;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree2(new pcl::search::KdTree<pcl::PointXYZ>);
    LRFDescriptorExtractor(cloud,keyPointsCloud,kdTree2,0.07,DescriptorOfSource);
    LRFDescriptorExtractor(cloudResult,keyPointCloudOfTarget,kdTree2,0.07,DescriptorOfTarget);

    std::cout <<" compute end" << std::endl;

    //std::cout << "******" << DescriptorOfSource.cols();
    //std::cout << "******" <<"\n" <<  DescriptorOfSource << std::endl;

    //note:此处是将用于测试的旋转+移动后的部分的点云重新计算回原来大点云的坐标下
    Eigen::Matrix<double,Eigen::Dynamic,3> match_result;
    computeDescriptorSimilarity(DescriptorOfTarget,DescriptorOfSource,0.01,match_result);

    Eigen::Matrix4d TranformMatrix;
    computeTransFromMatch(keyPointCloudOfTarget,keyPointsCloud,match_result,TranformMatrix);

    pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::transformPointCloud(*cloudResult,*matchCloudResult,TranformMatrix);

    std::cout << "the time of our process is:" << (clock()-time_start_our)/(double) CLOCKS_PER_SEC << "s" <<std::endl;

    //visualization2(cloudResult,keyPointCloudOfTarget,"keyPointsOfTarget");
    visualization2(cloud,keyPointsCloud,"keyPointsOfSource");

    computeTransformError(cloud_clip,matchCloudResult);
    computeTransformErrorByCoodinates(cloud_clip,matchCloudResult);
    visualization_correspondence(cloud,keyPointsCloud,cloudResult,keyPointCloudOfTarget,match_result);
    visualization2(cloud,matchCloudResult,"match_result_By_Our");


    //call our method end here
    //cloudResult 是剪裁后并进行旋转的结果，cloud是原点云，cloud_clip是剪裁但是没有旋转的点云，
    // 以下方法是找到cloudResult--------->cloud(通过keypoint)旋转矩阵，将cloudResult旋转回到cloud下对齐

    clock_t time_start_shot = clock();
    matchBySHOT(0.07,cloudResult,cloud,keyPointCloudOfTarget,keyPointsCloud,cloud_clip,matchCloudResult);
    std::cout << "the time of shot process is:" << (clock()-time_start_shot)/(double) CLOCKS_PER_SEC << "s" <<std::endl;

    clock_t time_start_fpfh = clock();
    matchByFPFH(0.07,cloudResult,cloud,keyPointCloudOfTarget,keyPointsCloud,cloud_clip,matchCloudResult);
    std::cout << "the time of fpfh process is:" << (clock()-time_start_fpfh)/(double) CLOCKS_PER_SEC << "s" <<std::endl;

    clock_t time_start_toldi = clock();
    matchByTOLDI(0.2,20,cloudResult,indices2,keyPointCloudOfTarget,
                 cloud,indices1,keyPointsCloud,cloud_clip,matchCloudResult);
    std::cout << "the time of toldi process is:" << (clock()-time_start_toldi)/(double) CLOCKS_PER_SEC << "s" <<std::endl;

    clock_t time_start_3dhopd = clock();
    matchBy3DHoPD(cloudResult,cloud,cloud_clip,cloud_clip,matchCloudResult);
    std::cout << "the time of 3dhopd process is:" << (clock()-time_start_3dhopd)/(double) CLOCKS_PER_SEC << "s" <<std::endl;

    return 0;
}