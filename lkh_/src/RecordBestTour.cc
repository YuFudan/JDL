#include "LKH.h"

/*
 * The RecordBestTour function records the current best tour in the BestTour
 * array.
 *
 * The function is called by LKHmain each time a run has resulted in a
 * shorter tour. Thus, when the predetermined number of runs have been
 * completed, BestTour contains an array representation of the best tour
 * found.
 */

void RecordBestTour() {
    if (ProblemType == TRPP) {
        int64_t *a = BestTour, *b = BetterTour;
        do {
            *a++ = *b++;
        } while (*b != -1);
    } else {
        for (int64_t i = 0; i <= Dim_salesman; i++)
            BestTour[i] = BetterTour[i];
    }
}
