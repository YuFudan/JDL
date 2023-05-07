#include "Heap.h"
#include "LKH.h"

/*
 * The AdjustClusters function adjusts K clusters with centers Center
 * so that they all have a size less than SubproblemSize.
 *
 * The function is called from KMeansClustering and KCenterClustering.
 */

void AdjustClusters(int64_t K, Node **Center) {
    int64_t d, i, j;
    Node *N;

    std::vector<int64_t> Size(K + 1);
    N = FirstNode;
    do
        Size[N->Subproblem]++;
    while ((N = N->Suc) != FirstNode);

    /* Remove nodes from clusters with size > SubproblemSize */
    for (i = 1; i <= K; i++) {
        if (Size[i] > SubproblemSize) {
            N = FirstNode;
            do {
                if (N->Subproblem == i) {
                    N->Rank = -N->Cost;
                    HeapLazyInsert(N);
                }
            } while ((N = N->Suc) != FirstNode);
            Heapify();
            for (j = 1; j <= SubproblemSize; j++)
                HeapDeleteMin();
            while ((N = HeapDeleteMin()))
                N->Subproblem = 0;
            Size[i] = SubproblemSize;
        }
    }
    /* Insert removed nodes into cluster with size < SubproblemSize */
    N = FirstNode;
    do {
        if (N->Subproblem == 0) {
            N->Cost = INT_MAX;
            j = 0;
            for (i = 1; i <= K; i++) {
                if (Size[i] < SubproblemSize &&
                    (d = Distance(N, Center[i])) < N->Cost) {
                    N->Cost = d;
                    j = i;
                }
            }
            Size[N->Subproblem = j]++;
        }
    } while ((N = N->Suc) != FirstNode);
}
