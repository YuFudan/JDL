#include "LKH.h"
#include "Segment.h"

GainType Penalty_TSPTW() {
    Node *N = Depot, *NextN = 0;
    GainType Sum = 0, P = 0;
    bool Forward = SUCC(N)->Id != N->Id + Dim_salesman;

    do {
        if (N->Id <= Dim_salesman) {
            if (Sum < N->Earliest)
                Sum = N->Earliest;
            else if (Sum > N->Latest &&
                     (P += Sum - N->Latest) > CurrentPenalty)
                return CurrentPenalty + 1;
            NextN = Forward ? SUCC(N) : PREDD(N);
        }
        NextN = Forward ? SUCC(NextN) : PREDD(NextN);
        Sum += N->CD[NextN->Index];
    } while ((N = NextN) != Depot);
    if (Sum > Depot->Latest &&
        ((P += Sum - Depot->Latest) > CurrentPenalty ||
         (P == CurrentPenalty && CurrentGain <= 0)))
        return CurrentPenalty + (CurrentGain > 0);
    return P;
}
