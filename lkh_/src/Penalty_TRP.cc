#include "LKH.h"
#include "Segment.h"

GainType Penalty_TRP() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType P = 0, DistanceSum;
    TruePenalty = 0;
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
                if (NextN->DepotId == 0) {
                    DistanceSum += N->CD[NextN->Index];
                    DistanceSum += NextN->ServiceTime;
                    P += DistanceSum;
                    if (P > CurrentPenalty ||
                        (P == CurrentPenalty && CurrentGain <= 0)) {
                        StartRoute = CurrentRoute;
                        return CurrentPenalty + (CurrentGain > 0);
                    }
                    if (DistanceSum > DistanceLimit) {
                        P += DistanceSum - DistanceLimit;
                        TruePenalty += DistanceSum - DistanceLimit;
                        if (P > CurrentPenalty ||
                            (P == CurrentPenalty && CurrentGain <= 0)) {
                            StartRoute = CurrentRoute;
                            return CurrentPenalty + (CurrentGain > 0);
                        }
                    }
                }
            }
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while ((N = NextN)->DepotId == 0);
    } while (N != StartRoute);
    return P;
}

GainType Penalty_TRPP() {
    Node *N = Depot, *Last = N;
    char Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;
    double t = 0, tt;
    double reward = 0;
    int64_t n;
    while (1) {
        N = Forward ? SUCC(N) : PREDD(N);
        n = N->Index;
        if (n == 0) {
            break;
        } else {
            tt = t + Last->CD[n];
            if (tt < Profits[n]) {
                reward += Profits[n] - tt;
                Last = N;
                t = tt;
            }
        }
    }
    return -reward;
}
