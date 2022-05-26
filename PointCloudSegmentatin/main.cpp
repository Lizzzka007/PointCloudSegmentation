#include <iostream>
#include<vector>
#include <fstream>
#include <string>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include "../Src/Geometry/Vector.h"
#include "../CUDAsrc/TripleClosestPoints/TripleClosestPoints.cuh"
#include <sys/time.h>

using namespace std;

#define Filename "Data/B-1843-0.987_precise_6041418.obj"

double time_now(void);
double time_now(void)
{
  struct timeval tv2;
  struct timezone tz;
  gettimeofday(&tv2, &tz);
  return tv2.tv_sec+tv2.tv_usec/1000000.0;
}

int main(void) 
{
    ifstream f (Filename);
    vector<VECTOR> points;
    vector<string> tokens;

    char c;
    string line;
    int step = 0;
    int N;

    if (f.is_open())
    {
        while( (getline(f,line)) && (line[0] == '#') )
        {
            // printf("step = %d\n", step);
            if(step == 7)
            {
                boost::split( tokens, line, boost::is_any_of(" ") );
                N = stoi( tokens[2] );
            }
            step++;
            continue;
        }

        N /= 100;

        for (int i = 0; i < N; i++)
        {
            VECTOR p;
            tokens.clear();

            (getline(f,line));

            boost::split( tokens, line, boost::is_any_of(" ") );

            // cout << tokens[0] << endl;

            p.x = stod(tokens[1]);
            p.y = stod(tokens[2]);
            p.z = stod(tokens[3]);

            points.push_back(p);

            (getline(f,line));
        }
        
    }
    else
    {
        printf("Can't open %s\n", Filename);
        return -1;
    }

    printf("Finish to read, N = %d\n", N);
    PontsTriple* Triple = new PontsTriple[N];

    for (int i = 0; i < N; i++)
    {
        for (int i = 0; i < Knearest; i++)
        {
            Triple[i].triangle[i] = VECTOR{0, 0, 0};
            Triple[i].triangleIds[i] = i;
            Triple[i].triangleVals[i] = 1e+10;
        }

        Triple[i].triangle[Knearest] = VECTOR{0, 0, 0};
    }

    double start = time_now();
    CalcPontsTriple<VECTOR, double>(points,Triple);
    double end = time_now();

    for (int i = 0; i < 20; i++)
    {
        Triple[i].triangle[0] = points[i];

        for (int j = 0; j < Knearest; j++)
        {
            // printf("%d / %d\n", j, Knearest);
            Triple[i].triangle[j + 1] = points[Triple[i].triangleIds[j]];
            printf("%e ", Triple[i].triangleVals[j]);
        }
        printf("\n");

        FILE* f = fopen(("T" + to_string(i) + ".txt").c_str(), "w");
        if(!f)
        {
            printf("Can't open file %s\n", ("T" + to_string(i) + ".txt").c_str());
            break;
        }

        printf("%d ", i);

        for (int j = 0; j < Knearest; j++)
        {
            printf("%d ", Triple[i].triangleIds[j]);
        }

        printf("\n");

        for (int j = 0; j < Knearest + 1; j++)
        {
            VECTOR v = Triple[i].triangle[j];
            printf("(%lf, %lf, %lf)\n", v.x, v.y, v.z);
            fprintf(f, "%lf %lf %lf\n", v.x, v.y, v.z);
        }

        fclose(f);

        printf("\n");
        
    }
    
    printf(">>> Elapsed time: %lf secs\n", end - start);

    delete[] Triple;
    tokens.clear();
    f.close();
    points.clear();
    return 0;
}