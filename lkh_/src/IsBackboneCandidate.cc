#include "LKH.h"

/*
 * The IsBackboneCandidate function is used to test if an edge, (ta,tb),
 * belongs to the set of backbone candidate edges.
 *
 * If the edge is a candidate edge the function returns 1; otherwise 0.
 */

bool IsBackboneCandidate(const Node *ta, const Node *tb) {
    for (auto *Nta = ta->BackboneCandidateSet; Nta && Nta->To; Nta++)
        if (Nta->To == tb)
            return true;
    return false;
}
