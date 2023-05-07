#include "LKH.h"
#include "Segment.h"

/*
 * The RecordBetterTour function is called by FindTour each time
 * the LinKernighan function has returned a better tour.
 *
 * The function records the tour in the BetterTour array and in the
 * BestSuc field of each node. Furthermore, for each node the previous
 * value of BestSuc is saved in the NextBestSuc field.
 *
 * Recording a better tour in the BetterTour array when the problem is
 * asymmetric requires special treatment since the number of nodes has
 * been doubled.
 */

void RecordBetterTour() {
    Node *N = Depot, *Stop = N;
    if (ProblemType == TRPP) {
        Node *N = Depot, *Last = N;
        int64_t i = 1;
        BetterTour[0] = 0;
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
                    BetterTour[i++] = n;
                    Last = N;
                    t = tt;
                }
            }
        }
        BetterTour[i] = -1;
    } else {
        if (!Asymmetric) {
            int64_t i = 1;
            do
                BetterTour[i++] = N->Id;
            while ((N = N->Suc) != Stop);
        } else {
            if (Stop->Id > Dim_salesman)
                Stop = N = Stop->Suc;
            if (N->Suc->Id != Dim_salesman + N->Id) {
                int64_t i = 1;
                do
                    if (N->Id <= Dim_salesman)
                        BetterTour[i++] = N->Id;
                while ((N = N->Suc) != Stop);
            } else {
                BetterTour[1] = N->Id;
                int64_t i = Dim_salesman;
                while ((N = N->Suc) != Stop)
                    if (N->Id <= Dim_salesman)
                        BetterTour[i--] = N->Id;
            }
        }
        BetterTour[0] = BetterTour[Dim_salesman];
    }
    N = FirstNode;
    do {
        N->NextBestSuc = N->BestSuc;
        N->BestSuc = N->Suc;
    } while ((N = N->Suc) != FirstNode);
}
