#include "LKH.h"

#undef max
static int64_t max(int64_t a, int64_t b) {
    return a > b ? a : b;
}

void PDPTW_Reduce() {
    int64_t i, j;
    Node *N, *V;

    if (Salesmen > 1)
        return;
    for (i = 1; i <= Dim_original; i++) {
        N = &NodeSet[i];
        for (j = 1; j <= Dim_original; j++) {
            if (j != i &&
                max(N->Earliest, N == Depot ? 0 : Depot->C[i]) +
                        N->ServiceTime + N->C[j] >
                    NodeSet[j].Latest)
                N->C[j] = BIG_INT;
        }
        if (N->Delivery) {
            for (j = 1; j < i; j++) {
                V = &NodeSet[j];
                if (V->Delivery && N->Demand + V->Demand > Capacity)
                    N->C[j] = V->C[i] = BIG_INT;
            }
        } else if (N->Pickup) {
            for (j = 1; j < i; j++) {
                V = &NodeSet[j];
                if (V->Pickup && -(N->Demand + V->Demand) > Capacity)
                    N->C[j] = V->C[i] = BIG_INT;
            }
        }
        if ((j = N->Pickup))
            NodeSet[i].C[j] = Depot->C[i] = BIG_INT;
        else if ((j = N->Delivery))
            NodeSet[j].C[i] = N->C[Depot->Id] = BIG_INT;
    }
}
