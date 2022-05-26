#pragma once
#include<iostream>
#include<vector>
// #include "../../Src/Geometry/Vector2.h"
#include "../../Src/Geometry/Vector.h"

#define Knearest 3

class PontsTriple
{
public:

    VECTOR* triangle = new VECTOR[Knearest + 1];
    int* triangleIds = new int[Knearest];
    double* triangleVals = new double[Knearest];
    std::vector<PontsTriple*> Neibour;

    PontsTriple()
    {
        for (int i = 0; i < Knearest; i++)
        {
            triangle[i] = VECTOR{0, 0, 0};
            triangleIds[i] = i;
            triangleVals[i] = 1e+10;
        }

        triangle[Knearest] = VECTOR{0, 0, 0};
    }

    ~PontsTriple()
    {
        delete[] triangle;
        delete[] triangleIds;
        delete[] triangleVals;
        Neibour.clear();
    }
};


template <typename DataType, typename T>
void CalcPontsTriple(const DataType* points, PontsTriple* Triple, const int pointCount);

template <typename DataType, typename T>
void CalcPontsTriple(std::vector<DataType>& points, PontsTriple* Triple);