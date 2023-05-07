#include "LKH.h"

/*
 * The CandidateReport function prints the minimum, average and maximum
 * number of candidates associated with a node.
 */

void CandidateReport() {
    int64_t Min = INT_MAX, Max = 0, Fixed = 0, Count;
    GainType Sum = 0, Cost = 0;
    Node *N;

    N = FirstNode;
    do {
        Count = 0;
        if (!N->CandidateSet.empty())
            for (auto *NN = N->CandidateSet.data(); NN->To; NN++)
                Count++;
        if (Count > Max)
            Max = Count;
        if (Count < Min)
            Min = Count;
        Sum += Count;
        if (N->FixedTo1 && N->Id < N->FixedTo1->Id) {
            Fixed++;
            Cost += Distance != Distance_1 ? Distance(N, N->FixedTo1) : 0;
        }
        if (N->FixedTo2 && N->Id < N->FixedTo2->Id) {
            Fixed++;
            Cost += Distance != Distance_1 ? Distance(N, N->FixedTo2) : 0;
        }
    } while ((N = N->Suc) != FirstNode);
    printff("Cand.min = %ld, Cand.avg = %0.1f, Cand.max = %ld\n",
            Min, (double)Sum / Dimension, Max);
    if (Fixed > 0)
        printff("Edges.fixed = %ld [Cost = " GainFormat "]\n", Fixed, Cost);
    if (MergeTourFiles >= 1) {
        Count = 0;
        N = FirstNode;
        do
            if (IsCommonEdge(N, N->MergeSuc[0]))
                Count++;
        while ((N = N->Suc) != FirstNode);
        printff("Edges.common = %ld\n", Count);
    }
}
