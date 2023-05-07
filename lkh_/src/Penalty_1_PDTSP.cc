#include "LKH.h"
#include "Segment.h"

GainType Penalty_1_PDTSP() {
    Node *N;
    GainType P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t Load = 0, MaxLoad = INT_MIN, MinLoad = INT_MAX;

    N = Depot;
    do {
        if (N->Id <= Dim_original) {
            Load += N->Demand;
            if (Load > MaxLoad)
                MaxLoad = Load;
            if (Load < MinLoad)
                MinLoad = Load;
            if (MaxLoad - MinLoad > Capacity) {
                P += MaxLoad - MinLoad - Capacity;
                if (P > CurrentPenalty ||
                    (P == CurrentPenalty && CurrentGain <= 0))
                    return CurrentPenalty + (CurrentGain > 0);
            }
        }
        N = Forward ? SUCC(N) : PREDD(N);
    } while (N != Depot);
    return P;
}
