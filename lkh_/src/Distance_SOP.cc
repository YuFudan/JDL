#include "LKH.h"

/*
 * The Distance_SOP function computes the distance for a SOP instance.
 */

int64_t Distance_SOP(Node* Na, Node* Nb) {
    int64_t d = OldDistance(Na, Nb);
    return d >= 0 ? d : MaxDistance;
}
