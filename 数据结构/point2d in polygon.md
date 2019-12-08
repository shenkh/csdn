# How can I determine whether a 2D Point is within a Polygon

https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html#The%20C%20Code

https://stackoverflow.com/a/2922778/9288778

```cpp
//int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy)
bool pnpoly(int nvert, vector<float> vertx, vector<float> verty, float testx, float testy)
{
  int i, j, c = 0;
  for (i = 0, j = nvert-1; i < nvert; j = i++) {
    if ( ((verty[i]>testy) != (verty[j]>testy)) &&
     (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
       c = !c;
  }
  return c;
}
```

Arguments

`nvert`: Number of vertices in the polygon. Whether to repeat the first vertex at the end has been discussed in the article referred above.
`vertx`, `verty`: Arrays containing the x- and y-coordinates of the polygon's vertices.
`testx`, `testy`: X- and y-coordinate of the test point.

It's both short and efficient and works both for convex and concave polygons. As suggested before, you should check the bounding rectangle first and treat polygon holes separately.

The idea behind this is pretty simple. The author describes it as follows:

I run a semi-infinite ray horizontally (increasing x, fixed y) out from the test point, and count how many edges it crosses. At each crossing, the ray switches between inside and outside. This is called the Jordan curve theorem.

The variable c is switching from 0 to 1 and 1 to 0 each time the horizontal ray crosses any edge. So basically it's keeping track of whether the number of edges crossed are even or odd. 0 means even and 1 means odd.

-----

```cpp
#include <iostream>
#include<vector>
using namespace std;

//https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html#The%20C%20Code
//无法处理点落在顶点上的情况
bool pnpoly(int nvert, vector<float> vertx, vector<float> verty, float testx, float testy)
{
    int i, j, c = 0;
    for (i = 0, j = nvert - 1; i < nvert; j = i++) {
        if (((verty[i] > testy) != (verty[j] > testy)) &&
            (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]))
            c = !c;
    }
    return c;
}

//https://mrkod.com/2014-04-19/point-in-polygon-ray-casting/index.html
bool pointInPolygon(int nvert, vector<float> polyX, vector<float> polyY, float x, float y) {

    int   i, j = nvert - 1;
    bool  oddNodes = false;

    for (i = 0; i < nvert; i++) {
        if ((polyY[i] < y && polyY[j] >= y
            || polyY[j] < y && polyY[i] >= y)
            && (polyX[i] <= x || polyX[j] <= x)) {
            oddNodes ^= (polyX[i] + (y - polyY[i]) / (polyY[j] - polyY[i]) * (polyX[j] - polyX[i]) < x);
        }
        j = i;
    }

    return oddNodes;
}

int main()
{
    int nvert;
    cin >> nvert;
    vector<float> vertx(nvert), verty(nvert);

    int i = 0;
    while (i < nvert) {
        cin >> vertx[i] >> verty[i];
        ++i;
    }

    cout << pnpoly(nvert, vertx, verty, 0.9, 1);
    cout << pointInPolygon(nvert, vertx, verty, 200, 1);
    cout << pointInPolygon(nvert, vertx, verty, 1.1, 1);
}

/*
5
1 0
1 200
200 200
200 1
100 100
*/

```