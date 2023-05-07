#include "LKH.h"
#include "Segment.h"

GainType Penalty_PDTSP() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *M, *CurrentRoute;
    ;
    GainType P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t Pickups = 0, Deliveries = 0;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    N = StartRoute;
    do {
        CurrentRoute = N;
        Serial++;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                if (N->Pickup) {
                    if (++Deliveries > Pickups ||
                        NodeSet[N->Pickup].Serial != Serial)
                        P++;
                } else if (N->Delivery) {
                    N->Serial = Serial;
                    if (++Pickups == 1) {
                        M = N;
                        while (M != Depot) {
                            if (M->Pickup)
                                M->Serial = Serial;
                            M = Forward ? SUCC(M) : PREDD(M);
                        }
                    }
                    if (NodeSet[N->Delivery].Serial != Serial)
                        P++;
                }
            }
            if (P > CurrentPenalty ||
                (P == CurrentPenalty && CurrentGain <= 0)) {
                StartRoute = CurrentRoute;
                return CurrentPenalty + (CurrentGain > 0);
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
    } while (N != StartRoute);
    return P;
}
