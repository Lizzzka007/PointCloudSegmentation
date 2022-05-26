#include "TripleClosestPoints.cuh"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <iostream>

#define mat_mult cublasDgemm
#define col_m(i,j,ld) (((j)*(ld))+(i))
using namespace std;

template <typename T>
__device__  __host__ void swap(T *arr, const int j, const int jp)
{
    T val;

    val = arr[j];
    arr[j] = arr[jp];
    arr[jp] = val;
}

template <typename T>
__device__ void devBubbleSort(int *DevBestIds, T *arr, const int n)
{
    int i, j;
    for (i = 0; i < n - 1; i++)
        for (j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
            {
                swap(arr, j, j + 1);
                swap(DevBestIds, j, j + 1);
            }
}

template <typename T>
__host__ void hostBubbleSort(int *DevBestIds, T *arr, const int n)
{
    int i, j;
    for (i = 0; i < n - 1; i++)
        for (j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
            {
                swap(arr, j, j + 1);
                swap(DevBestIds, j, j + 1);
            }

    // for (i = 0; i < n - 1; i++)
    //     printf("%d ", DevBestIds[i]);
    // printf("\n");
    // for (i = 0; i < n - 1; i++)
    //     printf("%e ", arr[i]);
    // printf("\n");printf("\n");
}

template <typename T>
__global__ void chooseNearestK(int *DevBestIds, T *DevVals, const T *DevDistanceMatrix, const int Dim, const int point_count, const int point_to_check_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
    T val;
    int ids[Knearest];
    int ids_next[Knearest + 1];
    T vals[Knearest];
    T vals_next[Knearest + 1];
    int insteadI;

    if(i < point_count)
    {
        for (int j = 0; j < Knearest; j++)
        {
            insteadI = j;
            if(insteadI == i)
                insteadI += Knearest;

            vals[j] = DevDistanceMatrix[col_m(i,insteadI,Dim)];
            ids[j] = insteadI;
        }

        devBubbleSort(ids, vals, Knearest);

        // if(i == 1)
        // {
        //     printf("Start\n");
        //     for (int s = 0; s < Knearest; s++)
        //     {
        //         printf("%d -> %e ", ids[s], vals[s]);
        //     }
        //     printf("\n");
        // }

        for (int j = 0; j < Knearest; j++)
        {
            vals_next[j] = vals[j];
            ids_next[j] = ids[j];
        }

        vals_next[Knearest] = 1e+10;
        ids_next[Knearest] = 100;
        

        for (int j = Knearest; j < i; j++)
        {
            val = DevDistanceMatrix[col_m(i,j,Dim)];
            
            vals_next[Knearest] = val;
            ids_next[Knearest] = j;

            devBubbleSort(ids_next, vals_next, Knearest + 1);
        }
        for (int j = i + 1; j < point_to_check_count; j++)
        {
            val = DevDistanceMatrix[col_m(i,j,Dim)];
            
            vals_next[Knearest] = val;
            ids_next[Knearest] = j;

            // if(i == 1)
            // {
            //     printf("Before\n");
            //     for (int s = 0; s < Knearest + 1; s++)
            //     {
            //         printf("%e ", vals_next[s]);
            //     }
            // }
            devBubbleSort(ids_next, vals_next, Knearest + 1);
            // if(i == 1)
            // {
            //     printf("\nAfter\n");
            //     for (int s = 0; s < Knearest + 1; s++)
            //     {
            //         printf("%e ", vals_next[s]);
            //     }
            //     printf("\n");
            //     printf("\n");
            // }
        }

        for (int j = 0; j < Knearest; j++)
        {
            DevVals[i* Knearest + j] = vals_next[j];
            DevBestIds[i* Knearest + j] = ids_next[j];
        }
    }

    // if(i == 0)
    // {
    //     printf("Here\n");
    //     for (int j = 0; j < Knearest; j++)
    //     {
    //         printf("%d ", DevBestIds[j]);
    //     }
    //     printf("\n");
    //     for (int j = 0; j < Knearest; j++)
    //     {
    //         printf("%e ", DevVals[j]);
    //     }
    //     printf("\n");
    //     printf("\n");
    // }
}

template <typename T>
__global__ void addSquares(T *DevDistanceMatrix, const T *p, const T *pT, const int Dim)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < Dim)
    {
        T squareI = p[col_m(i, 0, Dim)] * p[col_m(i, 0, Dim)] + p[col_m(i, 1, Dim)] * p[col_m(i, 1, Dim)] + p[col_m(i, 2, Dim)] * p[col_m(i, 2, Dim)];

        for (int j = 0; j < Dim; j++)
        {
            atomicAdd(DevDistanceMatrix + col_m(i,j,Dim), squareI);
            atomicAdd(DevDistanceMatrix + col_m(j,i,Dim), squareI);
        }

    }
}

template <typename T>
void PrintMatrix(T *a, const int rows, const int columns)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
            printf("%f ", (a[col_m(i, j, rows)]));
        printf("\n");   
    }
    printf("\n"); 
}

template <typename T>
void PrintDeviceMatrix(T *dev, const int rows, const int columns)
{
    T *host;
    host = (T*)malloc(rows * columns * sizeof(T));
    cudaMemcpy(host, dev, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
    PrintMatrix(host, rows, columns);
    free(host);
}

template <typename DataType, typename T>
void fillPoints(const std::vector<DataType>& points, T*pointsHost)
{
    const int pointCount = points.size();

    for (int i = 0; i < pointCount; i++)
    {
        pointsHost[col_m(i, 0, pointCount)] = points[i].x;
        pointsHost[col_m(i, 1, pointCount)] = points[i].y;
        pointsHost[col_m(i, 2, pointCount)] = points[i].z;
    }
    
}

template <typename DataType, typename T>
void CalcPontsTriple(std::vector<DataType>& points, PontsTriple* Triple)
{
    size_t free_mem, total;
    cudaMemGetInfo( &free_mem, &total );

    const int pointCount = points.size();
    const int doubleCount = int(float(free_mem) / sizeof(T));
    const int s = 6 * sizeof(T) + Knearest * (sizeof(int) + sizeof(T) );
    const int ss = sizeof(T);
    int pointBatchSize = floor((-s + sqrt(ss + 4 * ss * free_mem)) / (2.0 * ss));
    int *DevBestIds, *HostBestIds;

    const T alpha = -2.0;
    const T beta = 0.0;
    T *DevDistanceMatrix, *pT, *p, *pointsHost;
    T *DevBestVals, *HostBestVals;
    cudaError_t cuda_status;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t cuda_ret;

    if(pointBatchSize > 1000)
        pointBatchSize = (pointBatchSize / 1000) * 1000;
    if(pointBatchSize > pointCount)
        pointBatchSize = pointCount;

    const int batchCount = ceil(float(pointCount) / pointBatchSize);
    const int size_offset = 3 * pointBatchSize;
    int point_count_to_copyI, point_count_to_copyJ;
    printf("Batch count is %d, batch size is %d, avail double count is %d, count of points is %d\n", batchCount, pointBatchSize, doubleCount, pointCount);

    pointsHost = (T*) malloc(3 * pointCount * sizeof(T));
    if(!pointsHost)
    {
        printf("Cannot allocate memory for pointsHost\n");

        cublasDestroy(handle);

		return;
    }

    HostBestIds = (int*) malloc(Knearest * pointBatchSize * sizeof(int));
    if(!HostBestIds)
    {
        printf("Cannot allocate memory for pointsHost\n");

        free(pointsHost);

        cublasDestroy(handle);

		return;
    }

    HostBestVals = (T*) malloc(Knearest * pointBatchSize * sizeof(T));
    if(!HostBestIds)
    {
        printf("Cannot allocate memory for pointsHost\n");

        free(pointsHost);
        free(HostBestIds);

        cublasDestroy(handle);

		return;
    }


    cuda_status = cudaMalloc((void**)&(p), 3 * pointBatchSize * sizeof(T));
    if (cuda_status != cudaSuccess)
    {
        printf("Cannot allocate memory for pT\n");
        
        free(pointsHost);
        free(HostBestIds);
        free(HostBestVals);

        cublasDestroy(handle);
        return;
    }

    cuda_status = cudaMalloc((void**)&(pT), 3 * pointBatchSize * sizeof(T));
    if (cuda_status != cudaSuccess)
    {
        printf("Cannot allocate memory for pT\n");
        free(HostBestIds);
        free(pointsHost);
        free(HostBestVals);

        cudaFree(p);

        cublasDestroy(handle);
        return;
    }

    cuda_status = cudaMalloc((void**)&(DevDistanceMatrix), pointBatchSize * pointBatchSize * sizeof(T));
    if (cuda_status != cudaSuccess)
    {
        printf("Cannot allocate memory for pT\n");

        free(HostBestIds);
        free(pointsHost);
        free(HostBestVals);

        cudaFree(p);
        cudaFree(pT);

        cublasDestroy(handle);
        return;
    }

    cuda_status = cudaMalloc((void**)&(DevBestIds), pointBatchSize * Knearest * sizeof(int));
    if (cuda_status != cudaSuccess)
    {
        printf("Cannot allocate memory for DevBestIds\n");

        free(HostBestIds);
        free(pointsHost);
        free(HostBestVals);

        cudaFree(p);
        cudaFree(pT);
        cudaFree(DevDistanceMatrix);

        cublasDestroy(handle);
        return;
    }

    cuda_status = cudaMalloc((void**)&(DevBestVals), pointBatchSize * Knearest * sizeof(T));
    if (cuda_status != cudaSuccess)
    {
        printf("Cannot allocate memory for DevBestIds\n");

        free(HostBestIds);
        free(HostBestVals);
        free(pointsHost);

        cudaFree(p);
        cudaFree(pT);
        cudaFree(DevDistanceMatrix);

        cublasDestroy(handle);
        return;
    }

    fillPoints(points, pointsHost);
    points.clear();

    for (int i = 0; i < batchCount; i++)
    {
        point_count_to_copyI = (i < batchCount - 1) ? pointBatchSize : (pointCount - i * pointBatchSize);
        cuda_status = cudaMemset(p, 0, sizeof(T) * size_offset);
        printf("i: %d / %d\n", i, batchCount);
        if (cuda_status != cudaSuccess)
        {
            printf("Cannot do cudaMemset for p\n");
            cudaFree(DevDistanceMatrix);
            cudaFree(pT);
            cudaFree(p);
            cudaFree(DevBestVals);
            free(pointsHost);
            free(HostBestIds);
            free(HostBestVals);
            cublasDestroy(handle);
            return;
        }

        cuda_status = cudaMemcpy(p, pointsHost + i * size_offset, sizeof(T) * point_count_to_copyI * 3, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            printf("Cannot do cudaMemcpy for p\n");
            cudaFree(DevDistanceMatrix);
            cudaFree(pT);
            cudaFree(p);
            cudaFree(DevBestVals);
            free(pointsHost);
            free(HostBestIds);
            free(HostBestVals);
            cublasDestroy(handle);
            return;
        }

        for (int j = 0; j < batchCount; j++)
        {
            // printf("    j: %d / %d\n", j, batchCount);
            // printf("    point_count_to_copy = %d\n", point_count_to_copy);
            point_count_to_copyJ = (j < batchCount - 1) ? pointBatchSize : (pointCount - j * pointBatchSize);
            cuda_status = cudaMemset(pT, 0, sizeof(T) * size_offset);
            if (cuda_status != cudaSuccess)
            {
                printf("Cannot do cudaMemset for pT\n");
                cudaFree(DevDistanceMatrix);
                cudaFree(pT);
                cudaFree(p);
                cudaFree(DevBestVals);
                free(pointsHost);
                free(HostBestVals);
                free(HostBestIds);
                cublasDestroy(handle);
                return;
            }

            cuda_status = cudaMemcpy(pT, pointsHost + j * size_offset, sizeof(T) * point_count_to_copyJ * 3, cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess)
            {
                printf("Cannot do cudaMemcpy for pT\n");
                cudaFree(DevDistanceMatrix);
                cudaFree(pT);
                cudaFree(p);
                cudaFree(DevBestVals);
                free(pointsHost);
                free(HostBestIds);
                free(HostBestVals);
                cublasDestroy(handle);
                return;
            }

            cuda_status = cudaMemset(DevDistanceMatrix, 0, sizeof(T) * pointBatchSize * pointBatchSize);
            if (cuda_status != cudaSuccess)
            {
                printf("Cannot do cudaMemset for DevDistanceMatrix\n");
                cudaFree(DevDistanceMatrix);
                cudaFree(pT);
                cudaFree(p);
                cudaFree(DevBestVals);
                free(pointsHost);
                free(HostBestIds);
                free(HostBestVals);
                cublasDestroy(handle);
                return;
            }

            cuda_ret = mat_mult(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
            pointBatchSize, pointBatchSize, 3,
            &alpha,
            p, pointBatchSize,
            pT, pointBatchSize,
            &beta,
            DevDistanceMatrix, pointBatchSize);

            addSquares<<< ceil(float(pointBatchSize) / 32), 32  >>>(DevDistanceMatrix, p, pT, pointBatchSize);

            // PrintDeviceMatrix(DevDistanceMatrix, pointBatchSize, pointBatchSize);

            if (cuda_ret != CUBLAS_STATUS_SUCCESS) 
            {
                printf ("Cannot do cublasDgemm\n");
            
                cublasDestroy(handle);
                cudaFree(DevDistanceMatrix);
                cudaFree(pT);
                cudaFree(p);
                cudaFree(DevBestVals);
                free(HostBestIds);
                free(HostBestVals);
                free(pointsHost);
                return;
            }

            chooseNearestK<<< ceil(float(pointBatchSize) / 32), 32 >>>(DevBestIds, DevBestVals, DevDistanceMatrix, pointBatchSize, pointBatchSize, point_count_to_copyJ);

            // PrintDeviceMatrix(DevBestVals + 1, 1, Knearest);

            cuda_status = cudaMemcpy(HostBestIds, DevBestIds, pointBatchSize * Knearest * sizeof(int), cudaMemcpyDeviceToHost);
            if (cuda_status != cudaSuccess)
            {
                printf("Cannot do cudaMemcpy for pT\n");
                cudaFree(DevDistanceMatrix);
                cudaFree(pT);
                cudaFree(p);
                cudaFree(DevBestVals);
                free(pointsHost);
                free(HostBestIds);
                free(HostBestVals);
                cublasDestroy(handle);
                return;
            }

            cuda_status = cudaMemcpy(HostBestVals, DevBestVals, pointBatchSize * Knearest * sizeof(T), cudaMemcpyDeviceToHost);
            if (cuda_status != cudaSuccess)
            {
                printf("Cannot do cudaMemcpy for pT\n");
                cudaFree(DevDistanceMatrix);
                cudaFree(pT);
                cudaFree(p);
                cudaFree(DevBestVals);
                free(pointsHost);
                free(HostBestIds);
                free(HostBestVals);
                cublasDestroy(handle);
                return;
            }

            int compare_ids[Knearest * 2];
            T compare_vals[Knearest * 2];

            for (int s = 0; s < point_count_to_copyI; s++)
            {
                // printf("Here %d / %d\n", s, point_count_to_copy);
                mempcpy(compare_ids, HostBestIds + s * Knearest, sizeof(int) * Knearest);
                // printf("After1\n");
                mempcpy(compare_ids + Knearest, Triple[i * pointBatchSize + s].triangleIds, sizeof(int) * Knearest);
                // printf("After2\n");
                mempcpy(compare_vals, HostBestVals + s * Knearest, sizeof(T) * Knearest);
                // printf("After3\n");
                // for (int q = 0; q < Knearest; q++)
                // {
                //     printf("%e ", Triple[i * pointBatchSize + s].triangleVals[q]);
                // }
                // printf("\n");
                
                mempcpy(compare_vals + Knearest, Triple[i * pointBatchSize + s].triangleVals, sizeof(T) * Knearest);
                // printf("After4\n");
                hostBubbleSort(compare_ids, compare_vals, Knearest * 2);

                // if(i == 0)
                // {
                //     printf("i == 0\n");
                //     for (int a = 0; a < Knearest * 2 - 1; a++)
                //         printf("%d ", compare_ids[a]);
                //     printf("\n");
                //     for (int a = 0; a < Knearest * 2 - 1; a++)
                //         printf("%e ", compare_vals[a]);
                //     printf("\n");printf("\n");
                // }
                // printf("After5\n");
                mempcpy(Triple[i * pointBatchSize + s].triangleIds, compare_ids, sizeof(int) * Knearest);
                // printf("After6\n");
                mempcpy(Triple[i * pointBatchSize + s].triangleVals, compare_vals, sizeof(T) * Knearest);
                // printf("After7\n"); 

                // for (int t = 0; t < Knearest; t++)
                // {
                //     printf("%d ", Triple[i * pointBatchSize + s].triangleIds[t]);
                // }
                // printf("\n");
                // printf("End\n");
                
            }

            // PrintMatrix(T *a, const int rows, const int columns);
        }
        
    }
    
    
    cublasDestroy(handle);
    cudaFree(DevDistanceMatrix);
    cudaFree(DevBestVals);
    cudaFree(pT);
    cudaFree(p);
    free(pointsHost);
    free(HostBestIds);
    free(HostBestVals);
}

template void CalcPontsTriple<VECTOR, double>(std::vector<VECTOR>& points, PontsTriple* Triple);
