//
// Created by che on 2022/8/4.
//
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <Eigen/Eigenvalues>

#include "../include/LRFDsecriptorExtractor.h"
#include "../include/3DHoPD.h"
#include "../include/pointTools.h"
#include "../include/pointTransform.h"

void matchBy3DHoPD(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clip,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_resource,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr &matchCloudResult)

{
    //clock_t time_start = clock();

    pcl::PointCloud<pcl::PointXYZ> pointCloud1;
    pcl::PointCloud<pcl::PointXYZ> pointCloud2;
    for(pcl::PointCloud<pcl::PointXYZ>::iterator i=cloud1->begin();i<cloud1->end();i++){
        pointCloud1.push_back(*i);
    }

    for(pcl::PointCloud<pcl::PointXYZ>::iterator i=cloud2->begin();i<cloud2->end();i++){
        pointCloud2.push_back(*i);
    }

    cout<<"**********3DHoPD debugging information**********"<<endl;
    cout << pointCloud1.size() <<endl;


    threeDHoPD RP1, RP2;

    // Using Simple Uniform Keypoint Detection

    RP1.cloud = pointCloud1;
    RP1.detect_uniform_keypoints_on_cloud(0.1); //0.1
    cout << "Keypoints on Model: " << RP1.cloud_keypoints.size() << endl;

    RP2.cloud = pointCloud2;
    RP2.detect_uniform_keypoints_on_cloud(0.1); //0.1
    cout << "Keypoints on Scene: " << RP2.cloud_keypoints.size() << endl;


    // setup
    RP1.kdtree.setInputCloud(RP1.cloud.makeShared());// This is required for SUPER FAST
    RP2.kdtree.setInputCloud(RP2.cloud.makeShared());// THIS IS REQUIRED FOR SUPER FAST

    RP1.JUST_REFERENCE_FRAME_descriptors(0.1); //0.3
    RP2.JUST_REFERENCE_FRAME_descriptors(0.1);  //0.3

    pcl::Correspondences corrs;
    Eigen::Matrix<double,300,3> match_result;
    int match_num = 0 ;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_LRF;
    pcl::PointCloud<pcl::PointXYZ> pcd_LRF;
    for (int i = 0; i < RP2.cloud_LRF_descriptors.size(); i++)
    {
        pcl::PointXYZ point;
        point.x = RP2.cloud_LRF_descriptors[i].vector[0];
        point.y = RP2.cloud_LRF_descriptors[i].vector[1];
        point.z = RP2.cloud_LRF_descriptors[i].vector[2];

        pcd_LRF.push_back(point);
    }

    kdtree_LRF.setInputCloud(pcd_LRF.makeShared());
    for (int i = 0; i < RP1.cloud_LRF_descriptors.size(); i++)
    {
        pcl::PointXYZ searchPoint;
        searchPoint.x = RP1.cloud_LRF_descriptors[i].vector[0];
        searchPoint.y = RP1.cloud_LRF_descriptors[i].vector[1];
        searchPoint.z = RP1.cloud_LRF_descriptors[i].vector[2];

        std::vector<int> nn_indices;
        std::vector<float> nn_sqr_distances;

        std::vector<double> angles_vector;

        //0.3
        if (kdtree_LRF.radiusSearch(searchPoint,0.1,nn_indices,nn_sqr_distances) > 0)// IMPORTANT PARAMETER 0.2m or 0.1m ...
        {
            for (int j = 0; j < nn_indices.size(); j++)
            {
                //if (angle < 0.5)// Important Threshold!!!
                {
                    Eigen::VectorXf vec1, vec2;
                    vec1.resize(15); vec2.resize(15);

                    for (int k = 0; k < 15; k++)
                    {
                        vec1[k] = RP1.cloud_distance_histogram_descriptors[i].vector[k];
                        vec2[k] = RP2.cloud_distance_histogram_descriptors[nn_indices[j]].vector[k];

                    }

                    double dist = (vec1-vec2).norm();
                    //add program
                    if(dist < 0.4)  //0.2
                        angles_vector.push_back(dist);

                    //angles_vector.push_back(dist);

                }
            }

            if(!(angles_vector.begin() == angles_vector.end())){

                std::vector<double>::iterator result;
                result = std::min_element(angles_vector.begin(), angles_vector.end());
                //std::cout << "Max element at: " << std::distance(match_distance.begin(), result) << '\n';
                //std::cout << "Max element is: " << match_distance[std::distance(match_distance.begin(), result)] << '\n';
                int min_element_index = std::distance(angles_vector.begin(), result);

                match_result.row(match_num)[0] = i;
                match_result.row(match_num)[1] = nn_indices[min_element_index];
                match_result.row(match_num)[2] = angles_vector[min_element_index];
                match_num++;
            }

            /*pcl::Correspondence corr;
            corr.index_query = RP1.patch_descriptor_indices[i];// vulnerable
            corr.index_match = RP2.patch_descriptor_indices[nn_indices[min_element_index]];// vulnerable

            corrs.push_back(corr);*/

        }
    }


    match_result.block(0,0,match_num,3);

    //cout << "No. of Reciprocal Correspondences : " << corrs.size() << endl;
    /*pcl::CorrespondencesConstPtr corrs_const_ptr = boost::make_shared< pcl::Correspondences >(corrs);

    pcl::Correspondences corr_shot;
    pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection_shot;
    Ransac_based_Rejection_shot.setInputSource(RP1.cloud_keypoints.makeShared());
    Ransac_based_Rejection_shot.setInputTarget(RP2.cloud_keypoints.makeShared());
    Ransac_based_Rejection_shot.setInlierThreshold(0.05);
    Ransac_based_Rejection_shot.setInputCorrespondences(corrs_const_ptr);
    Ransac_based_Rejection_shot.getCorrespondences(corr_shot);*/


    //cout << "Transformation Matrix : \n" << Ransac_based_Rejection_shot.getBestTransformation()<< endl;

    //cout << "True correspondences after RANSAC : " << corr_shot.size() << endl;

    Eigen::Matrix4d TransforMatrix;
    //Eigen::Matrix4f TransforMatrix = Ransac_based_Rejection_shot.getBestTransformation();
    //pcl::PointCloud<pcl::PointXYZ>::Ptr matchCloudResult(new pcl::PointCloud<pcl::PointXYZ>);
    computeTransFromMatch(RP1.cloud_keypoints.makeShared(),RP2.cloud_keypoints.makeShared(),match_result,TransforMatrix);
    pcl::transformPointCloud(*cloud_resource,*matchCloudResult,TransforMatrix);
   // cout << "the time of 3DHoPD process is:" << (clock()-time_start) / (double) CLOCKS_PER_SEC << "ms" <<endl;
    visualization2(cloud2,matchCloudResult,"match_result_by_3DHoPD");

    //computeTransformError(cloud_clip,matchCloudResult);
    computeTransformErrorByCoodinates(cloud_clip,matchCloudResult);


}
