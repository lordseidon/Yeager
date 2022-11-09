#include <iostream>
#include <map>
#include <bits/stdc++.h>


using namespace std;

constexpr int  MVV_LVA[15][15] = {
        {0, 0, 0, 0, 0,  0, 0, 0, 100,  33,  33,  20,  11,  1, 0},           // victim P, attacker p, n, b, r, q, k, None
        {0, 0, 0, 0, 0,  0, 0, 0, 300,  100,  100,  60, 33, 3, 0},           // victim N, attacker p, n, b, r, q, k, None
        {0, 0, 0, 0, 0,  0, 0, 0, 300,  100,  100,  60, 33, 3, 0},          // victim B, attacker p, n, b, r, q, k, None
        {0, 0, 0, 0, 0,  0, 0, 0, 500,  166,  166, 100, 56, 6, 0},       // victim R, attacker p, n, b, r, q, k, None
        {0, 0, 0, 0, 0,  0, 0, 0, 900, 300, 300, 180, 100, 10, 0},          // victim Q, attacker p, n, b, r, q, k, None
        {0,  0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0},       // victim K, attacker p, n, b, r, q, k, None
        {0,  0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0},       // vacant
        {0,  0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0},       // vacant
        {100,  33,  33,  20,  11,  1, 0, 0, 0, 0, 0, 0,  0, 0, 0},        // victim P, attacker p, n, b, r, q, k, None
        {300,  100,  100,  60, 33, 3, 0, 0, 0, 0, 0, 0,  0, 0, 0},       // victim N, attacker p, n, b, r, q, k, None
        {300,  100,  100,  60, 33, 3, 0, 0, 0, 0, 0, 0,  0, 0, 0},      // victim B, attacker p, n, b, r, q, k, None
        {500,  166,  166, 100, 56, 6, 0, 0, 0, 0, 0, 0,  0, 0, 0},      // victim r, attacker p, n, b, r, q, k, None
        {900, 300, 300, 180, 100, 10, 0, 0, 0, 0, 0, 0,  0, 0, 0},      // victim Q, attacker p, n, b, r, q, k, None
        {0,  0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0},       // victim K, attacker p, n, b, r, q, k, None
        {0,  0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0},       // victim None, attacker p, n, b, r, q, k, None
};


int partition(Move arr[], int start, int end, Position board)
{
 
    int pivot = MVV_LVA[board.at(arr[start].to())][board.at(arr[start].from())];
 
    int count = 0;
    for (int i = start + 1; i <= end; i++) {
        if (MVV_LVA[board.at(arr[i].to())][board.at(arr[i].from())] >= pivot)
            count++;
    }
 
    // Giving pivot element its correct position
    int pivotIndex = start + count;
    swap(arr[pivotIndex], arr[start]);
 
    // Sorting left and right parts of the pivot element
    int i = start, j = end;
 
    while (i < pivotIndex && j > pivotIndex) {
 
        while (MVV_LVA[board.at(arr[i].to())][board.at(arr[i].from())] >= pivot) {
            i++;
        }
 
        while (MVV_LVA[board.at(arr[j].to())][board.at(arr[j].from())] < pivot) {
            j--;
        }
 
        if (i < pivotIndex && j > pivotIndex) {
            swap(arr[i++], arr[j--]);
        }
    }
 
    return pivotIndex;
}
 
void quickSort(Move arr[], int start, int end, Position board)
{
 
    // base case
    if (start >= end)
        return;
 
    // partitioning the array
    int p = partition(arr, start, end, board);
 
    // Sorting the left part
    quickSort(arr, start, p - 1, board);
 
    // Sorting the right part
    quickSort(arr, p + 1, end, board);
}
