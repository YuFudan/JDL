#include "LKH.h"

void VRPB_Reduce() {
    int64_t i, j, n = Dim_original;

    for (i = 1; i <= n; i++) {
        if (NodeSet[i].Backhaul) {
            for (j = 1; j <= n; j++)
                if (j != i && j != MTSPDepot && !NodeSet[j].Backhaul)
                    NodeSet[i].C[j] = BIG_INT;
            NodeSet[MTSPDepot].C[i] = BIG_INT;
        }
    }
}
