#include "LKH.h"

void TSPTW_Reduce() {
    int64_t i, j, k, n = Dim_salesman;
    char **R;

    if (Salesmen > 1)
        return;
    R = new char *[n + 1]();
    for (i = 1; i <= n; i++)
        R[i] = new char[n + 1]();
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            if (j != i &&
                NodeSet[j].Earliest + NodeSet[j].C[i] > NodeSet[i].Latest)
                R[i][j] = 1;
    /* Compute the transitive closure */
    for (k = 1; k <= n; k++)
        for (i = 1; i <= n; i++)
            if (R[i][k])
                for (j = 1; j <= n; j++)
                    R[i][j] |= R[k][j];
    /* Eliminate edges */
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            if (j != i && R[i][j])
                NodeSet[j].C[i] = BIG_INT;
    /* Generate constraints */
    for (i = 1; i <= n; i++) {
        Node *Ni = &NodeSet[i];
        for (j = 1; j <= n; j++) {
            if (i != j && R[i][j]) {
                Node *Nj = &NodeSet[j];
                Constraint *Con;
                Con = new Constraint();
                Con->t1 = Ni;
                Con->t2 = Nj;
                Con->Suc = FirstConstraint;
                FirstConstraint = Con;
                Con->Next = Ni->FirstConstraint;
                Ni->FirstConstraint = Con;
            }
        }
    }
    for (i = 1; i <= n; i++)
        free(R[i]);
    free(R);
}
