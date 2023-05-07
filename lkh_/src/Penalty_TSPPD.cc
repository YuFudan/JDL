#include "LKH.h"
#include "Segment.h"

GainType Penalty_TSPPD() {
    Node *N;
    GainType P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t Load = Capacity;

    N = Depot;
    do {
        if (N->Id <= Dim_original && N != Depot) {
            Load += N->Demand;
            if (Load > Capacity)
                P += Load > Capacity;
            if (Load < 0)
                P -= Load;
            if (P > CurrentPenalty ||
                (P == CurrentPenalty && CurrentGain <= 0))
                return CurrentPenalty + (CurrentGain > 0);
        }
        N = Forward ? SUCC(N) : PREDD(N);
    } while (N != Depot);
    return P;
}
