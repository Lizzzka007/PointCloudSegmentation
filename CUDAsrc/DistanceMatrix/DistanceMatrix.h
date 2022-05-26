#pragma once
#include<iostream>
#include<vector>

template <typename DataType, typename T>
void CalcDistanceMatrix(const DataType* points, T* DistanceMatrix, const int pointCount);

template <typename DataType, typename T>
void CalcDistanceMatrix(const std::vector<DataType>& points, T* DistanceMatrix);