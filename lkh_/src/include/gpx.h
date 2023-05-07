#ifndef _GPX_H
#define _GPX_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "GainType.h"
#include "LKH.h"

#define new_int(n) (new int64_t[n]())
#define new_tour(n) (new tour[n]())
#define new_gate_structure(n) (new gate_structure[n]())

typedef struct Adj {
    int64_t vertex;
    struct Adj *nextAdj;
} Adj;

typedef struct Graph {
    int64_t numVertices;
    Adj **firstAdj, **lastAdj;
} Graph;

Graph *new_Graph(int64_t n);
void insertEdge(Graph *g, int64_t v1, int64_t v2);
void freeGraph(Graph *g);
void compCon(Graph *g, int64_t *vector_comp);

typedef struct {
    int64_t num;
    int64_t time;
} gate_structure;

typedef struct {
    gate_structure *inputs;
    gate_structure *outputs;
    gate_structure first_entry;
    gate_structure last_exit;
    GainType fitness;
} tour;

GainType gpx(int64_t *solution_blue, int64_t *solution_red, int64_t *offspring);
void new_candidates(int64_t *vector_comp, int64_t n_new);
void free_candidates(void);
void findInputs(int64_t *sol_blue, int64_t *sol_red);
void testComp(int64_t cand);
int64_t testUnfeasibleComp(int64_t *sol_blue);
void fusion(int64_t *sol_blue, int64_t *sol_red);
void fusionB(int64_t *sol_blue, int64_t *sol_red);
void fusionB_v2(int64_t *sol_blue, int64_t *sol_red);
GainType off_gen(int64_t *sol_blue, int64_t *sol_red, int64_t *offspring,
                 int64_t *label_list);

extern int64_t n_cities, n_cand;
extern int64_t n_partitions_size2, n_partitions_before_fusion,
    n_partitions_after_fusion1, n_partitions_after_fusion2,
    n_partitions_after_fusion3;
extern int64_t n_partitions_after_fusion4, n_partitions_after_fusion5,
    n_partitions_after_fusionB;
extern Node **Map2Node;

int64_t *alloc_vectori(int64_t lines);
int64_t **alloc_matrix(int64_t lines, int64_t collums);
void dealloc_matrix(int64_t **Matrix, int64_t lines);
int64_t weight(int64_t i, int64_t j);
int64_t d4_vertices_id(int64_t *solution_blue, int64_t *solution_red,
                   int64_t *d4_vertices, int64_t *common_edges_blue,
                   int64_t *common_edges_red);
void insert_ghost(int64_t *solution, int64_t *solution_p2, int64_t *d4_vertices,
                  int64_t *label_list_inv);
void tourTable(int64_t *solution_blue_p2, int64_t *solution_red_p2,
               int64_t *solution_red, int64_t *label_list, int64_t *label_list_inv,
               int64_t *vector_comp, int64_t n_new, int64_t *common_edges_blue_p2,
               int64_t *common_edges_red_p2);

#endif
