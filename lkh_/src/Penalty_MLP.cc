#include "LKH.h"
#include "Segment.h"

GainType Penalty_MLP() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType P = 0, DistanceSum;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    N = StartRoute;
    do {
        CurrentRoute = N;
        DistanceSum = 0;
        do {
            NextN = Forward ? SUCC(N) : PREDD(N);
            if (N->Id <= Dim_original || N->DepotId) {
                DistanceSum += N->CD[NextN->Index];
                DistanceSum += NextN->ServiceTime;
                P += DistanceSum;
                if (P > CurrentPenalty ||
                    (P == CurrentPenalty && CurrentGain <= 0)) {
                    StartRoute = CurrentRoute;
                    return CurrentPenalty + (CurrentGain > 0);
                }
                if (DistanceSum > DistanceLimit &&
                    ((P += DistanceSum - DistanceLimit) > CurrentPenalty ||
                     (P == CurrentPenalty && CurrentGain <= 0))) {
                    StartRoute = CurrentRoute;
                    return CurrentPenalty + (CurrentGain > 0);
                }
            }
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while ((N = NextN)->DepotId == 0);
    } while (N != StartRoute);
    return P;
}
