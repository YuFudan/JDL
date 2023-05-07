#include "LKH.h"
#include "Segment.h"

GainType TSPTW_MakespanCost() {
    Node *N = Depot, *NextN;
    GainType Sum = 0;
    bool Forward = SUCC(N)->Id != N->Id + Dim_salesman;

    if (ProblemType != TSPTW)
        return 0;
    do {
        if (N->Id <= Dim_salesman)
            if (Sum < N->Earliest)
                Sum = N->Earliest;
        NextN = Forward ? SUCC(N) : PREDD(N);
        Sum += N->CD[NextN->Index];
    } while ((N = NextN) != Depot);
    return Sum;
}
