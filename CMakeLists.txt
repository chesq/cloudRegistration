cmake_minimum_required(VERSION 2.8)
project(pointCloudRegistration)
find_package(PCL REQUIRED)
find_package(Eigen3)
find_package(OpenCV)

if(Opencv_FOUND)
    message(STATUS "The Opecv lib is found!")
endif()

include_directories(${PCL_INCLUDE_DIRS} ${EIGEN_INCLUDE_LIBS} ${OpenCV_INCLUDE_DIRS})

add_executable(pointRegistration src/test2.cpp src/pointTransform.cpp src/pointTools.cpp
        src/clippingAndFilter.cpp src/registrateByICP.cpp src/LRFDEscriptorExtractor.cpp
        src/matchBySHOT.cpp src/matchByFPFH.cpp src/matchByTOLDI.cpp src/TOLDI_LRF.cpp src/TOLDI_descriptor.cpp
        src/matchBy3DHoPD.cpp)

target_link_libraries(pointRegistration ${PCL_LIBRARIES} ${OpenCV_LIBS})
