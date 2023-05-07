#include <algorithm>
#include <tuple>

#include "LKH.h"

/*
 * The CreateNNCandidateSet function creates for each node
 * a candidate set consisting of the K least costly neighbor edges.
 *
 * Time complexity is O(K*N*N).
 *
 * The function is called from the CreateCandidateSet and
 * AddExtraCandidates functions. It is intended for use on non-geometric
 * and toroidal instances.
 */

void CreateNNCandidateSet(int64_t K) {
    Node *Na, *Nb;
    int64_t d, a, b, k, Count, Forward;

    if (TraceLevel >= 2)
        printff("Creating NN candidate set ... \n");
    std::vector<Node *> XNearList(Dimension), To(K + 1);
    std::vector<int64_t> Cost(K + 1);
    for (Na = FirstNode, k = 0; k < Dimension; Na = Na->Suc, k++)
        XNearList[k] = Na;
    std::sort(XNearList.begin(), XNearList.end(), [](const Node *a, const Node *b) {
        return std::tie(a->X, a->Y) < std::tie(b->X, b->Y);
    });
    for (a = 0; a < Dimension; a++) {
        Na = XNearList[a];
        Count = 0;
        for (Forward = 0; Forward <= 1; Forward++) {
            b = Forward ? a + 1 : a - 1;
            while (b >= 0 && b < Dimension) {
                Nb = XNearList[b];
                d = Distance(Na, Nb);
                k = Count < K ? Count++ : K;
                while (k > 0 && d < Cost[k - 1]) {
                    To[k] = To[k - 1];
                    Cost[k] = Cost[k - 1];
                    k--;
                }
                To[k] = Nb;
                Cost[k] = d;
                b = Forward ? b + 1 : b - 1;
            }
        }
        for (k = 0; k < Count; k++)
            AddCandidate(Na, To[k], D(Na, To[k]), 2);
    }
    if (TraceLevel >= 2)
        printff("done\n");
}