#include "LKH.h"
#include "Segment.h"

static int64_t *Queue;

GainType Penalty_PDTSPF() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t QueueLength, Front, Back, Capacity = Dim_original;

    if (!Queue)
        Queue = new int64_t[Capacity]();
    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    N = StartRoute;
    do {
        CurrentRoute = N;
        QueueLength = Front = 0;
        Back = -1;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                if (N->Pickup) {
                    if (QueueLength > 0) {
                        if (Queue[Front] != N->Id)
                            P++;
                        Front = (Front + 1) % Capacity;
                        QueueLength--;
                    } else
                        P++;
                    if (P > CurrentPenalty ||
                        (P == CurrentPenalty && CurrentGain <= 0)) {
                        StartRoute = CurrentRoute;
                        return CurrentPenalty + (CurrentGain > 0);
                    }
                } else if (N->Delivery) {
                    Back = (Back + 1) % Capacity;
                    Queue[Back] = N->Delivery;
                    QueueLength++;
                }
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
        P += QueueLength;
    } while (N != StartRoute);
    return P;
}
