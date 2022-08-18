//
// Created by che on 2022/7/29.
//
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <Eigen/Eigenvalues>
#include <vector>
#include <pcl/common/transforms.h>

#include "../include/LRFDsecriptorExtractor.h"

void LRFDescriptorExtractor(pcl::PointCloud<pcl::PointXYZ>::Ptr &inputCloud,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &inputKeyCloud,
                            pcl::search::KdTree<pcl::PointXYZ>::Ptr &kdTree,
                            double searchRadius,
                            Eigen::Matrix<double,32*3,Eigen::Dynamic> &Descriptor){

    kdTree->setInputCloud(inputCloud);
    std::vector<int> pointID2radiusSearch;
    std::vector<float> pointSearchDist;
    size_t input_cloud_size = inputKeyCloud->size();
    Descriptor.resize(32*3,input_cloud_size);
    pcl::PointXYZ keyPoint;
    pcl::PointCloud<pcl::PointXYZ>::Ptr searchCloud(new pcl::PointCloud<pcl::PointXYZ>);
    const int MaxClassfication = 32;


    for(size_t i = 0;i<input_cloud_size;i++){

        searchCloud->clear();
        keyPoint = inputKeyCloud->points[i];
        if(kdTree->radiusSearch(keyPoint,searchRadius,pointID2radiusSearch,pointSearchDist) > 0){

            for(int j=0;j<pointID2radiusSearch.size();j++)
                searchCloud->push_back(inputCloud->points[pointID2radiusSearch[j]]);
        }

        Eigen::Vector4d centroid;
        Eigen::Matrix3d convariance;
        Eigen::Matrix3d eigenVectorMarix;
        Eigen::Vector3d eigenValues;
        Eigen::Vector3d xNormal;
        Eigen::Vector3d yNormal;
        Eigen::Vector3d zNormal;

        pcl::compute3DCentroid(*searchCloud,centroid);
        pcl::computeCovarianceMatrixNormalized(*searchCloud,centroid,convariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(convariance,Eigen::ComputeEigenvectors);
        eigenVectorMarix = eigen_solver.eigenvectors();
        eigenValues = eigen_solver.eigenvalues();

        //Computation eliminates the ambiguity of xnoraml
        double multSum;
        Eigen::Vector3d sumNormalComputor;
        xNormal = Eigen::Vector3d(eigenVectorMarix(0),eigenVectorMarix(1),eigenVectorMarix(2));

        for (int k = 0; k < searchCloud->size() ; ++k) {

            sumNormalComputor.x() = searchCloud->points[k].x - keyPoint.x;
            sumNormalComputor.y() = searchCloud->points[k].y - keyPoint.y;
            sumNormalComputor.y() = searchCloud->points[k].z - keyPoint.z;

            multSum += sumNormalComputor.dot(xNormal);

        }

        if(multSum < 0)
            xNormal = -xNormal;

        /*if(i == n-1){
            std::cout << "normal:" << xNormal.x() << "," <<xNormal.y() << "," << xNormal.z() << std::endl;

            std::cout << "eigenValues:" << eigenValues.x() << "," <<eigenValues.y() << "," << eigenValues.z() << std::endl;

            // from the print test, we can know the eigenValues listed from low to high clearly; 22.7.30---che
        }*/
        xNormal.normalize();
        Eigen::Vector3d tempVector;
        //compute the yNormal,zNormal;
        tempVector.x() = keyPoint.x - centroid.x();
        tempVector.y() = keyPoint.y - centroid.y();
        tempVector.z() = keyPoint.z - centroid.z();

        yNormal = tempVector - tempVector.dot(xNormal) * xNormal;
        yNormal.normalize();

        zNormal = xNormal.cross(yNormal);
        zNormal.normalize();

        /*if(i == n-1) {
            std::cout << "************the fowlloing are debugging information of normal computation************"<<std::endl;
            std::cout << "the ambiguity of xnoraml compute result is:" << multSum << std::endl;
            std::cout << "xnormal(" << xNormal.x() << "," << xNormal.y() << "," << xNormal.z() << ")" << std::endl;

            std::cout << "ynormal:(" << yNormal.x() << "," << yNormal.y() << "," << yNormal.z() << ")" << std::endl;
            std::cout << "znormal:(" << zNormal.x() << "," << zNormal.y() << "," << zNormal.z() << ")" << std::endl;
            std::cout << "eigenValues:" << eigenValues.x() << "," << eigenValues.y() << "," << eigenValues.z()
                      << std::endl;
            std::cout << "************************************************" << std::endl;

            // from the print test, we can know the eigenValues listed from low to high clearly; 22.7.30---che
        }*/

        //Desdcriptor computation
        // Encouraged by the S.H.O.T idea,j presents radial number and k for projection quadrant, l for position number
        int j,k,l,indexOfPoint;
        pcl::PointCloud<pcl::PointXYZL>::Ptr classficationPointCloud(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::PointXYZL classPoint;

        for(int i = 0;i<searchCloud->size();i++){

            Eigen::Vector3d vectorParamOfPoint;
            vectorParamOfPoint.x() = searchCloud->points[i].x - keyPoint.x;
            vectorParamOfPoint.y() = searchCloud->points[i].y - keyPoint.y;
            vectorParamOfPoint.z() = searchCloud->points[i].z - keyPoint.z;
            double r = vectorParamOfPoint.norm();
            double zProjection = vectorParamOfPoint.dot(xNormal);

            if(r < searchRadius  && zProjection > 0)
                j = 0;
            else if (r > searchRadius  && zProjection > 0)
                j = 1;
            else if (r < searchRadius  && zProjection < 0)
                j = 2;
            else
                j = 3;

            //为了表达清楚，这里使用中文：求出区域内的某个点在yoz平面内的投影向量（以关键点为圆心），同时注意归一化，
            // 再求投影向量与y,z轴的方向投影，能够判断在平面内的哪个区间（用投影的正负来区分），再求与y轴的夹角，判断落在哪个区域，
            // 此处夹角用 在z轴上的投影 比上 在y轴的投影（即tan Alpha）与tan45=1比较,
            //注意编号方式：从第一象限到第四象限，每个象限分为2个区域，分别编号1-8；

            Eigen::Vector3d vectorProjectionOfPoint;
            vectorProjectionOfPoint = vectorParamOfPoint - vectorParamOfPoint.dot(xNormal) * xNormal;
            vectorProjectionOfPoint.normalize();
            if(vectorProjectionOfPoint.dot(yNormal)>0 && vectorProjectionOfPoint.dot(zNormal)>0)
                k = 0;  //the first quadrant
            else if(vectorProjectionOfPoint.dot(yNormal)<0 && vectorProjectionOfPoint.dot(zNormal)>0)
                k = 1;
            else if (vectorProjectionOfPoint.dot(yNormal)<0 && vectorProjectionOfPoint.dot(zNormal)<0)
                k = 2;
            else
                k = 3;

            if((abs(vectorProjectionOfPoint.dot(zNormal)) / abs(vectorProjectionOfPoint.dot(yNormal)))<=1)
                l = 0;
            else
                l = 1;

            // seve the searchpoint and it's label;

            indexOfPoint = j*8 + k*2 + l; //indexOfPoint valued from 0 to 31;
            classPoint.label = indexOfPoint;
            classPoint.x = searchCloud->points[i].x;
            classPoint.y = searchCloud->points[i].y;
            classPoint.z = searchCloud->points[i].z;

            classficationPointCloud->push_back(classPoint);


        }
        /*if(i == 0){
            std::cout << "*************classficationpointclod*************" << std::endl;
            std::cout << classficationPointCloud->points[1] << std::endl;
        }*/
        //compute descriptor,,l_index------>label index
        int class_n = classficationPointCloud->size();
        pcl::PointCloud<pcl::PointXYZ>::Ptr classPoint_compute(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Vector4d centroid_class;
        Eigen::Vector3d feature_class_cord;
        Eigen::Vector3d feature_value;
        Eigen::VectorXd feature_of_keypoint = Eigen::VectorXd(32*3);  //VectorXd need initial.
        for(int l_index=0;l_index<MaxClassfication;l_index++){
            classPoint_compute->clear();

            for(int j = 0; j<class_n;j++){
                if(classficationPointCloud->points[j].label == l_index)
                    classPoint_compute->push_back(searchCloud->points[j]);
            }


            if(classPoint_compute->size() > 0){
                pcl::compute3DCentroid(*classPoint_compute,centroid_class);
                feature_class_cord.x() = centroid_class.x() - keyPoint.x;
                feature_class_cord.y() = centroid_class.y() - keyPoint.y;
                feature_class_cord.z() = centroid_class.z() - keyPoint.z;

                feature_class_cord.normalize();
                double wight = double(classPoint_compute->size()) / double(class_n);  //compute the wight

                feature_value.x() = wight * feature_class_cord.dot(xNormal);
                feature_value.y() = wight * feature_class_cord.dot(yNormal);
                feature_value.z() = wight * feature_class_cord.dot(zNormal);

            }

            else{
                feature_value.x()=0;
                feature_value.y()=0;
                feature_value.z()=0;
            }

            feature_of_keypoint(l_index*3+0) = feature_value.x();
            feature_of_keypoint(l_index*3+1) = feature_value.y();
            feature_of_keypoint(l_index*3+2) = feature_value.z();

            //std::cout << "done" <<std::endl;

        }


//        std::cout << "**************test**************" <<std::endl;
//        std::cout << feature_of_keypoint <<std::endl;
//        std::cout << "**************end**************" <<std::endl;
//        std::cout << feature_of_keypoint.size() <<std::endl;
        //Descriptor.col(i) = feature_of_keypoint;   //this do not work!!
        //Descriptor.resize(32*3,i+1);     //!!!!!!!!!! remember this step
        //*********take care of this bug***********
        //*********take care of this bug***********
        //*********take care of this bug***********

        Descriptor.col(i) =  feature_of_keypoint;
        //std::cout << "**********class_test***********" << Descriptor.col(0)<<std::endl;
    }

    //std::cout << "********the following is the descriptor compute debugging information********" << std::endl;
    //std::cout << "the descriptor size of the input cloud is:" <<std::endl;
    //std::cout <<"(" << Descriptor.rows() <<"," << Descriptor.cols() << ")" << std::endl;
}

void computeDescriptorSimilarity(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfSource,
                                 Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfTarget,
                                 double matchThreshold,Eigen::Matrix<double,Eigen::Dynamic,3> &match_result){

    double similarityTemp;
    double similarityMin;
    int targetIndex;
    bool matchSuccess;
    int match_rows;
    int match_index = 0;
    Eigen::Matrix<double,Eigen::Dynamic,3> match;

    match_rows = DescriptorOfSource.cols();
    match.resize(match_rows,3);
    match.setZero();

    for(int i=0;i<DescriptorOfSource.cols();++i){
        similarityMin = 1e3;
        matchSuccess = false;
        for (int j = 0; j <DescriptorOfTarget.cols();++j) {
            similarityTemp = (DescriptorOfSource.col(i)-DescriptorOfTarget.col(j)).norm();
            //std::cout << "similarity value:" << similarityTemp << std::endl;
            if(similarityTemp < similarityMin && similarityTemp < matchThreshold){
                similarityMin = similarityTemp;
                targetIndex = j;
                matchSuccess = true;
            }
        }

        if(matchSuccess){

            //std::cout << "*******test match*******"<<i<<","<<targetIndex<<","<<similarityMin<<std::endl;
            match.row(match_index)(0) = i;
            match.row(match_index)(1) = targetIndex;
            match.row(match_index)(2) = similarityMin;
            match_index++;
        }
    }


    match_result.resize(match_index,3);
    match_result = match.block(0,0,match_index,3);

    std::cout<<"**********the following are the descriptor match debugging information**********" <<std::endl;
    std::cout<<"match_result size: " << match_result.rows() << std::endl;
    //std::cout << match_result << std::endl;

}

void computeDescriptorSimilarity_biside(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfSource,
                                 Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> DescriptorOfTarget,
                                 double matchThreshold,Eigen::Matrix<double,Eigen::Dynamic,3> &match_result){

    double similarityTemp,similarityTemp2;
    double similarityMin,similarityMin2;
    int targetIndex,targetIndex2;
    bool match1Success,match2Success;
    int match_rows = 1000;
    int match_index = 0;
    Eigen::Matrix<double,Eigen::Dynamic,3> match;
    match.resize(match_rows,3);
    match.setZero();

    for(int i=0;i<DescriptorOfSource.cols();++i){
        similarityMin = 1e3;
        similarityMin2 = 1e3;
        match1Success = false,match2Success = false;
        for (int j = 0; j <DescriptorOfTarget.cols();++j) {
            similarityTemp = (DescriptorOfSource.col(i)-DescriptorOfTarget.col(j)).norm();
            //std::cout << "similarity value:" << similarityTemp << std::endl;
            if(similarityTemp < similarityMin && similarityTemp < matchThreshold){
                match1Success = true;
                similarityMin = similarityTemp;
                targetIndex = j;
            }
        }

        if(match1Success){
            for(int k=0;k<DescriptorOfSource.cols();k++){

                similarityTemp2 = (DescriptorOfSource.col(k)-DescriptorOfTarget.col(targetIndex)).norm();

                if(similarityTemp2 < similarityMin2 && similarityTemp2 < matchThreshold){
                    similarityMin2 = similarityTemp2;
                    targetIndex2 = k;
                }
            }

        }

        if( i == targetIndex2)
            match2Success = true;

        if(match2Success){

            //std::cout << "*******test match*******"<<i<<","<<targetIndex<<","<<similarityMin<<std::endl;
            match.row(match_index)(0) = i;
            match.row(match_index)(1) = targetIndex;
            match.row(match_index)(2) = similarityMin;
            match_index++;
            //remove the matched descriptor in DescriptorOfTarget
            DescriptorOfTarget.block(0,targetIndex,DescriptorOfTarget.rows(),DescriptorOfTarget.cols()-1-targetIndex) =
                    DescriptorOfTarget.block(0,targetIndex+1,DescriptorOfTarget.rows(),DescriptorOfTarget.cols()-1-targetIndex);
            DescriptorOfTarget.conservativeResize(DescriptorOfTarget.rows(),DescriptorOfTarget.cols()-1);

            match2Success = false;
            match1Success = false;

        }
    }


    match_result.resize(match_index,3);
    match_result = match.block(0,0,match_index,3);

    std::cout<<"**********the following are the descriptor match debugging information**********" <<std::endl;
    std::cout<<"match_result size: " << match_result.rows() << std::endl;
    //std::cout << match_result << std::endl;

}

void computeTransFromMatch(pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud,
                           Eigen::Matrix<double,Eigen::Dynamic,3> match_result,
                           Eigen::Matrix4d &TranformMatrix){

    int match_long;
    Eigen::Matrix<double,3,Eigen::Dynamic> sourcePoints,targetPoints;

    match_long = match_result.rows();
    sourcePoints.resize(3,match_long);
    targetPoints.resize(3,match_long);


    for (int i = 0; i < match_long; i++) {
        sourcePoints(0,i) = sourceCloud->points[match_result.row(i)(0)].x;
        sourcePoints(1,i) = sourceCloud->points[match_result.row(i)(0)].y;
        sourcePoints(2,i) = sourceCloud->points[match_result.row(i)(0)].z;

        targetPoints(0,i) = targetCloud->points[match_result.row(i)(1)].x;
        targetPoints(1,i) = targetCloud->points[match_result.row(i)(1)].y;
        targetPoints(2,i) = targetCloud->points[match_result.row(i)(1)].z;
    }

    /*std::cout << "done" << std::endl;
    std::cout << sourcePoints.rows() << "," <<  sourcePoints.cols() << std::endl;
    std::cout << targetPoints.rows() << "," <<  targetPoints.cols() << std::endl;*/


    TranformMatrix = Eigen::umeyama(sourcePoints,targetPoints, false);

    //std::cout << "***********match transfrom debugging information***********" <<std::endl;
    std::cout <<TranformMatrix<<std::endl;

}
