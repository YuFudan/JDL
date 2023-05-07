#include "LKH.h"

/*
 * The AddCandidate function adds a given edge (From, To) to the set
 * of candidate edges associated with the node From. The cost and
 * alpha-value of the edge are passed as parameters to the function.
 *
 * The function has no effect if the edge is already in the candidate
 * set.
 *
 * If the edge was added, the function returns 1; otherwise 0.
 *
 * The function is called from the functions CreateDelaunaySet and
 * OrderCandidateSet.
 */

bool AddCandidate(Node *From, Node *To, int64_t Cost, int64_t Alpha) {
    if (From->Subproblem != FirstNode->Subproblem ||
        To->Subproblem != FirstNode->Subproblem ||
        Cost == INT_MAX)
        return false;
    if (From->CandidateSet.empty())
        From->CandidateSet.resize(3);
    if (From == To || To->Subproblem != FirstNode->Subproblem ||
        !IsPossibleCandidate(From, To))
        return false;

    int64_t Count = 0;
    Candidate *NFrom;
    for (NFrom = From->CandidateSet.data(); NFrom->To && NFrom->To != To; NFrom++)
        Count++;
    if (NFrom->To) {
        if (NFrom->Alpha == INT_MAX)
            NFrom->Alpha = Alpha;
        return false;
    }
    NFrom->Cost = Cost;
    NFrom->Alpha = Alpha;
    NFrom->To = To;
    From->CandidateSet.resize(Count + 2);
    From->CandidateSet[Count + 1].To = 0;
    return true;
}
