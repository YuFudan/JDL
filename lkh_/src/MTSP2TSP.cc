#include "Heap.h"
#include "LKH.h"

void MTSP2TSP() {
    Node *N = 0;
    int64_t i, AnyFixed = 0;

    if (Salesmen >= Dimension)
        eprintf("SALESMEN >= DIMENSION");
    if (SubproblemSize > 0)
        eprintf("SUBPROBLEM_SIZE > 0: Not implemented for %s instances",
                ProblemType);
    if (MTSPMaxSize < 0)
        MTSPMaxSize = Dimension - 1;
    if (MTSPMinSize < 0)
        MTSPMinSize =
            Dimension / ((1 + (Dimension - 1) / MTSPMaxSize) + 1);
    if (MTSPMinSize > MTSPMaxSize)
        eprintf("MTSP_MIN_SIZE > MTSP_MAX_SIZE");
    if (MTSPDepot > Dim_salesman)
        eprintf("DEPOT > DIMENSION");
    if (ProblemType == TSP || ProblemType == ATSP) {
        if (MTSPObjective == MINSUM)
            Penalty = Penalty_MTSP_MINSUM;
        else if (MTSPObjective == MINMAX)
            Penalty = Penalty_MTSP_MINMAX;
        else if (MTSPObjective == MINMAX_SIZE)
            Penalty = Penalty_MTSP_MINMAX_SIZE;
    } else if (ProblemType == CVRP)
        Penalty = Penalty_CVRP;
    if (ProblemType == CVRP || ProblemType == CTSP || ProblemType == TSP) {
        int64_t NewDimension = Dimension + Salesmen - 1;
        Node *Prev = 0;
        Node *OldNodeSet = NodeSet.data();
        NodeSet.resize(1 + NewDimension);
        Dim_original = NewDimension - Salesmen + 1;
        for (i = 1; i <= Dim_original; i++) {
            N = &NodeSet[i];
            if (N->FixedTo1) {
                N->FixedTo1 += NodeSet.data() - OldNodeSet;
                AnyFixed = 1;
            }
            if (N->FixedTo2)
                N->FixedTo2 += NodeSet.data() - OldNodeSet;
        }
        Depot = &NodeSet[MTSPDepot];
        Depot->Color = 0;
        FirstNode = &NodeSet[1];
        for (i = 1; i <= NewDimension; i++, Prev = N) {
            N = &NodeSet[i];
            if (i > Dimension) {
                *N = *Depot;
                N->FixedTo1 = N->FixedTo2 = 0;
                ASSERT(N->Id == i);
                if (MergeTourFiles >= 1)
                    N->MergeSuc =
                        new Node *[MergeTourFiles]();
            }
            if (i == 1)
                FirstNode = N;
            else
                Link(Prev, N);
            N->Special = 0;
        }
        Link(N, FirstNode);
        if (MergeTourFiles >= 1) {
            for (i = Dimension + 1; i <= NewDimension; i++) {
                N = &NodeSet[i];
                N->MergeSuc =
                    new Node *[MergeTourFiles]();
            }
        }
        Dimension = Dim_salesman = NewDimension;
        if (ProblemType != CTSP && Salesmen <= Dim_original && MTSPMinSize > 0 &&
            !AnyFixed) {
            HeapMake(Dim_original - 1);
            for (i = 1; i <= Dim_original; i++) {
                N = &NodeSet[i];
                if (N == Depot)
                    continue;
                N->Rank = Distance(N, Depot);
                HeapLazyInsert(N);
            }
            Heapify();
            for (i = 1; i <= Salesmen; i++)
                HeapDeleteMin()->Special = i;
            HeapClear();
            free(Heap);
            Heap = 0;
        }
    }
    for (i = Dim_original + 1; i <= Dim_salesman; i++) {
        NodeSet[i].Earliest = Depot->Earliest;
        NodeSet[i].Latest = Depot->Latest;
        NodeSet[i].Demand = Depot->Demand;
    }
    OldDistance = Distance;
    Distance = Distance_MTSP;
    WeightType = SPECIAL;
}
