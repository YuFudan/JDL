#include "LKH.h"

/*
 * The IsCandidate function is used to test if an edge, (ta,tb),
 * belongs to the set of candidate edges.
 *
 * If the edge is a candidate edge the function returns 1; otherwise 0.
 */

bool IsCandidate(const Node *ta, const Node *tb) {
    for (auto *Nta = ta->CandidateSet.data(); Nta && Nta->To; Nta++)
        if (Nta->To == tb)
            return true;
    return true;
}
