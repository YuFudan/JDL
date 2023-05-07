#ifndef _BIT_H
#define _BIT_H

/*
 * This header specifies the interface for the use of Binay Indexed Trees.
 */

#include "LKH.h"

void BIT_Make(int64_t Size);
void BIT_Update();

int64_t BIT_LoadDiff3Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                     Node *t5, Node *t6);
int64_t BIT_LoadDiff4Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                     Node *t5, Node *t6, Node *t7, Node *t8);
int64_t BIT_LoadDiff5Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                     Node *t5, Node *t6, Node *t7, Node *t8,
                     Node *t9, Node *t10, int64_t Case10);
int64_t BIT_LoadDiff6Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                     Node *t5, Node *t6, Node *t7, Node *t8,
                     Node *t9, Node *t10, Node *t11, Node *t12);

#endif
