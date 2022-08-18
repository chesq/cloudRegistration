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

    std::vector<std::string> file_name;
    std::string file_path = "../data/testdata/";
    std::string our_write_path = "../data/rmse/our_rmse.txt";
    std::string shot_write_path = "../data/rmse/shot_rmse.txt";
    std::string fpfh_write_path = "../data/rmse/fpfh_rmse.txt";
    std::string toldi_write_path = "../data/rmse/toldi_rmse.txt";
    std::string tdhopd_write_path = "../data/rmse/tdhopd_rmse.txt";

    getFileName(file_path,file_name);
    std::cout << "******debugging information******" << std::endl;
    Eigen::VectorXd our_dis_vector,shot_dis_vector,fpfh_dis_vector,toldi_dis_vector,tdhopd_dis_vector;
    int file_size = file_name.size();
    our_dis_vector.resize(file_size);
    shot_dis_vector.resize(file_size);
    fpfh_dis_vector.resize(file_size);
    toldi_dis_vector.resize(file_size);
    tdhopd_dis_vector.resize(file_size);
    double our_dis,shot_dis,fpfh_dis,toldi_dis,tdhopd_dis;

    clock_t time_start = clock();
    for(int i=0;i<file_size;i++){

        //std::string file_path = "../data/testdata/" + file_name[i];
        std::string file_path = "../data/testdata/airplane_00017.txt" ;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip(new pcl::PointCloud<pcl::PointXYZ>);
        createCloudFromTxt(file_path,cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudResult;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudResult_icp;
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_tran;
        Eigen::Matrix3d rotationMatrix;

        Eigen::Vector3d translationVector (0.01,0.1,0.3);//set the RT matrix for test
        //Eigen::Quaterniond q1(1,0.1,0.4,0.7);  //set the RT matrix for test
        Eigen::Quaterniond q1(0.4,0.4,0.7,0.07);
        q1.normalize();
        rotationMatrix = q1.toRotationMatrix();
        clipCloud(cloud,cloud_clip);

        cloudResult = cloudTransform(cloud_clip,rotationMatrix,translationVector);
        //***********iss3d detector

        pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointsCloud(new  pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloudOfTarget(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree1(new pcl::search::KdTree<pcl::PointXYZ>);
        std::vector<int> indices1;
        std::vector<int> indices2;

        iss_detector(cloud,keyPointsCloud,indices1,kdTree1);
        iss_detector(cloudResult,keyPointCloudOfTarget,indices2,kdTree1);

        //local reference frame-based descriptor extraction of the keypoints
        //compute the normal and the descriptor
        Eigen::Matrix<double,32*3,Eigen::Dynamic> DescriptorOfSource;
        Eigen::Matrix<double,32*3,Eigen::Dynamic> DescriptorOfTarget;

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree2(new pcl::search::KdTree<pcl::PointXYZ>);
        LRFDescriptorExtractor(cloud,keyPointsCloud,kdTree2,0.07,DescriptorOfSource);

        LRFDescriptorExtractor(cloudResult,keyPointCloudOfTarget,kdTree2,
                0.07,DescriptorOfTarget);

        //note:此处是将用于测试的旋转+移动后的部分的点云重新计算回原来大点云的坐标下
        Eigen::Matrix<double,Eigen::Dynamic,3> match_result;
        computeDescriptorSimilarity(DescriptorOfTarget,DescriptorOfSource,0.01,match_result);

        pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult_our(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult_shot(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult_fpfh(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult_toldi(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult_tdhopd(new pcl::PointCloud<pcl::PointXYZ>);


        //*******our method compute process
        Eigen::Matrix4d TranformMatrix;
        computeTransFromMatch(keyPointCloudOfTarget,keyPointsCloud,match_result,TranformMatrix);
        pcl::transformPointCloud(*cloudResult,*matchCloudResult_our,TranformMatrix);

        visualization2(cloud,matchCloudResult_our,"match_result_By_Our");

        //computeTransformError(cloud_clip,matchCloudResult_our);
        our_dis = computeTransformErrorByCoodinates(cloud_clip,matchCloudResult_our);
        our_dis_vector[i] = our_dis;

        //*****our method end



        /*matchBySHOT(0.07,cloudResult,cloud,keyPointCloudOfTarget,keyPointsCloud,cloud_clip,matchCloudResult_shot);
        //our_dis = computeTransformErrorByCoodinates(cloud_clip,matchCloudResult_shot);
        //our_dis_vector[i] = our_dis;*/

        /*matchByFPFH(0.07,cloudResult,cloud,keyPointCloudOfTarget,keyPointsCloud,cloud_clip,matchCloudResult_fpfh);
        our_dis = computeTransformErrorByCoodinates(cloud_clip,matchCloudResult_fpfh);
        our_dis_vector[i] = our_dis;*/

        /*matchByTOLDI(0.07,20,cloudResult,indices2,keyPointCloudOfTarget,
                     cloud,indices1,keyPointsCloud,cloud_clip,matchCloudResult_toldi);
        our_dis = computeTransformErrorByCoodinates(cloud_clip,matchCloudResult_toldi);
        our_dis_vector[i] = our_dis;*/

        /*matchBy3DHoPD(cloudResult,cloud,cloud_clip,matchCloudResult_tdhopd);
        our_dis = computeTransformErrorByCoodinates(cloud_clip,matchCloudResult_tdhopd);
        our_dis_vector[i] = our_dis;*/

        std:: cout << "pointcloud" << i << "done" << std::endl;
    }

    writeCloudToTxt(our_dis_vector,our_write_path);
    std::cout << "the time of our process is:" << (clock()-time_start)/(double) CLOCKS_PER_SEC << "s" <<std::endl;


    return 0;

}