#include "LKH.h"

/*
 * Each time a trial has resulted in a shorter tour the candidate set is
 * adjusted (by AdjustCandidateSet). The ResetCandidates function resets
 * the candidate set. The original order is re-established (using, and
 * edges with Alpha == INT_MAX are excluded.
 *
 * The function is called from FindTour and OrderCandidates.
 */

void ResetCandidateSet() {
    Node *From;
    Candidate *NFrom, *NN, Temp;

    From = FirstNode;
    /* Loop for all nodes */
    do {
        if (From->CandidateSet.empty())
            continue;
        auto *start = From->CandidateSet.data();
        /* Reorder the candidate array of From */
        for (NFrom = start; NFrom->To; NFrom++) {
            Temp = *NFrom;
            for (NN = NFrom - 1;
                 NN >= start &&
                 (Temp.Alpha < NN->Alpha ||
                  (Temp.Alpha == NN->Alpha && Temp.Cost < NN->Cost));
                 NN--)
                *(NN + 1) = *NN;
            *(NN + 1) = Temp;
        }
        NFrom--;
        /* Remove included edges */
        while (NFrom >= start + 2 && NFrom->Alpha == INT_MAX)
            NFrom--;
        NFrom++;
        NFrom->To = 0;
        /* Remove impossible candidates */
        for (NFrom = start; NFrom->To; NFrom++) {
            if (!IsPossibleCandidate(From, NFrom->To)) {
                for (NN = NFrom; NN->To; NN++)
                    *NN = *(NN + 1);
                NFrom--;
            }
        }
    } while ((From = From->Suc) != FirstNode);
}
