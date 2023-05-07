#include "LKH.h"
#include "Segment.h"

GainType Penalty_BWTSP() {
    Node *N = &NodeSet[1], *NextN, *CurrentNode;
    bool Forward = SUCC(N)->Id != N->Id + Dim_salesman;
    GainType Cost, P = 0;
    int64_t WhiteCount;
    static Node *StartNode = 0;

    if (!StartNode)
        StartNode = N;
    if (StartNode->Id > Dim_salesman)
        StartNode -= Dim_salesman;
    N = StartNode;
    do {
        CurrentNode = N;
        NextN = Forward ? SUCC(N) : PREDD(N);
        if (NextN->Id > Dim_salesman)
            NextN = Forward ? SUCC(NextN) : PREDD(NextN);
        WhiteCount = 0;
        Cost = 0;
        while (NextN->Id > BWTSP_B) {
            WhiteCount++;
            Cost += N->CD[NextN->Index];
            NextN = Forward ? SUCC(NextN) : PREDD(NextN);
            if (NextN->Id > Dim_salesman)
                NextN = Forward ? SUCC(NextN) : PREDD(NextN);
            N = NextN;
        }
        N = NextN;
        if (Cost > BWTSP_L)
            P += Cost - BWTSP_L;
        if (WhiteCount > BWTSP_Q)
            P += WhiteCount - BWTSP_Q;
        if (P > CurrentPenalty ||
            (P == CurrentPenalty && CurrentGain <= 0)) {
            StartNode = CurrentNode;
            return CurrentPenalty + (CurrentGain > 0);
        }
    } while (N != StartNode);
    return P;
}
