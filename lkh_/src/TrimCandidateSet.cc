#include "LKH.h"

/*
 * The TrimCandidateSet function takes care that each node has
 * associated at most MaxCandidates candidate edges.
 */

void TrimCandidateSet(int64_t MaxCandidates) {
    Node *From;
    Candidate *NFrom;
    int64_t Count, MaxDepotCandidates, MaxCand;

    MaxDepotCandidates = Dimension == Dim_salesman ? Salesmen : 2 * Salesmen;
    if (MaxDepotCandidates < MaxCandidates)
        MaxDepotCandidates = MaxCandidates;
    From = FirstNode;
    do {
        MaxCand = From->DepotId == 0 ? MaxCandidates : MaxDepotCandidates;
        Count = 0;
        for (NFrom = From->CandidateSet.data(); NFrom && NFrom->To; NFrom++)
            Count++;
        if (Count > MaxCand) {
            From->CandidateSet.resize(MaxCand + 1);
            From->CandidateSet[MaxCand].To = 0;
        }
    } while ((From = From->Suc) != FirstNode);
}
