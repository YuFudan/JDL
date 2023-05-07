#include "LKH.h"

/*
 * The IsCommonEdge function is used to test if an edge, (ta,tb),
 * is common to the set of tours to be merged.
 *
 * If the edge is a common edge the function returns 1; otherwise 0.
 */

bool IsCommonEdge(const Node* ta, const Node* tb) {
    if (MergeTourFiles < 2)
        return false;
    for (int64_t i = 0; i < MergeTourFiles; i++)
        if (ta->MergeSuc[i] != tb && tb->MergeSuc[i] != ta)
            return false;
    return true;
}
