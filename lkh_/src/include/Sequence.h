#ifndef SEQUENCE_H
#define SEQUENCE_H

/*
 * This header specifies the interface for the use of node sequences.
 *
 * The functions BestKOptMove and BacktrackKOptMove are implemented
 * by means of such sequences.
 */

#include "LKH.h"

extern Node **t;      /* The sequence of nodes to be used in a move */
extern Node **T;      /* The currently best t's */
extern Node **tSaved; /* For saving t when using the BacktrackKOptMove
                         function */
extern int64_t *p;        /* The permutation corresponding to the sequence in
                         which the t's occur on the tour */
extern int64_t *q;        /* The inverse permutation of p */
extern int64_t *incl;     /* Array: incl[i] == j, if (t[i], t[j]) is an
                         inclusion edge */
extern int64_t *cycle;    /* Array: cycle[i] is cycle number of t[i] */
extern GainType *G;   /* For storing the G-values in the BestKOptMove
                         function */
extern int64_t K;         /* The value K for the current K-opt move */

int64_t FeasibleKOptMove(int64_t k);
void FindPermutation(int64_t k);
int64_t Cycles(int64_t k);

int64_t Added(const Node *ta, const Node *tb);
int64_t Deleted(const Node *ta, const Node *tb);

void MarkAdded(Node *ta, Node *tb);
void MarkDeleted(Node *ta, Node *tb);
void UnmarkAdded(Node *ta, Node *tb);
void UnmarkDeleted(Node *ta, Node *tb);

#endif
