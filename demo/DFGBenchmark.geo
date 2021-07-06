// mesh of the DFG benchmark

// constants
sqrt1_2 = 0.70710678118654752440;

// geometry parameters
xc = 2.0; // position of the cylinder
yc = 2.0; // position of the cylinder
rc = 0.5; // radius of the cylinder
h = 4.1;  // height of the channel
l = 22.0; // length of the channel
a = 3.0;  // scaling factor

// characteristic length
cl__max = 10.0;
cl__med = h / 15.; //changed from / 10.
cl__min = rc / 2.;

// curves of the channel
Point(1) = {0, 0, 0, cl__max};
Point(2) = {xc, 0, 0, cl__max};
Point(3) = {a * xc, 0, 0, cl__max};
Point(4) = {l, 0, 0, cl__max};
Point(5) = {l, yc, 0, cl__max};
Point(6) = {l, h, 0, cl__max};
Point(7) = {a * xc, h, 0,cl__max};
Point(8) = {xc, h, 0, cl__max};
Point(9) = {0, h, 0, cl__max};
Point(10) = {0, yc, 0, cl__max};

// lines of the channel
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 1};

// points of the cylinder
Point(11) = {xc, yc, 0, cl__max};
Point(12) = {xc + rc, yc, 0, cl__max};
Point(13) = {xc + rc * sqrt1_2, yc + rc * sqrt1_2, 0, cl__max};
Point(14) = {xc, yc + rc, 0, cl__max};
Point(15) = {xc - rc * sqrt1_2, yc + rc * sqrt1_2, 0, cl__max};
Point(16) = {xc - rc, yc, 0, cl__max};
Point(17) = {xc - rc * sqrt1_2, yc - rc * sqrt1_2, 0, cl__max};
Point(18) = {xc, yc - rc, 0, cl__max};
Point(19) = {xc + rc * sqrt1_2, yc - rc * sqrt1_2, 0, cl__max};

// curves of the cylinder
Circle(11) = {12, 11, 13};
Circle(12) = {13, 11, 14};
Circle(13) = {14, 11, 15};
Circle(14) = {15, 11, 16};
Circle(15) = {16, 11, 17};
Circle(16) = {17, 11, 18};
Circle(17) = {18, 11, 19};
Circle(18) = {19, 11, 12};

// auxiliary horizontal lines
Point(20) = {a * xc, yc, 0, cl__max};
Line(19) = {10, 16};
Line(20) = {12, 20};
Line(21) = {20, 5};

// auxiliary vertical lines
Line(22) = {2, 18};
Line(23) = {14, 8};
Line(24) = {3, 20};
Line(25) = {20, 7};

// auxiliary diagonal lines
Line(26) = {1, 17};
Line(27) = {19, 3};
Line(28) = {9, 15};
Line(29) = {13, 7};

// surface definitions
Line Loop(30) = {1, 22, -16, -26};
Plane Surface(31) = {30};
Line Loop(32) = {2, -27, -17, -22};
Plane Surface(33) = {32};
Line Loop(34) = {3, 4, -21, -24};
Plane Surface(35) = {34};
Line Loop(36) = {5, 6, -25, 21};
Plane Surface(37) = {36};
Line Loop(38) = {7, -23, -12, 29};
Plane Surface(39) = {38};
Line Loop(40) = {23, 8, 28, -13};
Plane Surface(41) = {40};
Line Loop(42) = {9, 19, -14, -28};
Plane Surface(43) = {42};
Line Loop(44) = {10, 26, -15, -19};
Plane Surface(45) = {44};
Line Loop(46) = {11, 29, -25, -20};
Plane Surface(47) = {46};
Line Loop(48) = {18, 20, -24, -27};
Plane Surface(49) = {48};

// physical regions
Physical Curve("inlet", 100) = {9, 10};
Physical Curve("outlet", 101) = {4, 5};
Physical Curve("lower wall", 102) = {1, 2, 3};
Physical Curve("upper wall", 103) = {6, 7, 8};
Physical Curve("cylinder", 104) = {11, 12, 13, 14, 15, 16, 17, 18};
Physical Surface("fluid", 200) = {31, 33, 35, 37, 39, 41, 43, 45, 47, 49};

// definition of size fields
// box behind the cylinder
Field[1] = Box;
Field[1].VIn = cl__med;
Field[1].VOut = cl__max;
Field[1].XMax = l;
Field[1].XMin = a * xc;
Field[1].YMin = 0.0;
Field[1].YMax = h;
// box in front of the cylinder
Field[2] = Box;
Field[2].VIn = cl__min / 2; //changed from cl__min
Field[2].VOut = cl__max;
Field[2].XMax = a * xc;
Field[2].XMin = 0.0;
Field[2].YMin = 0.0;
Field[2].YMax = h;
// refinement of the channel walls and the cylinder boundary
Field[3] = Distance;
Field[3].EdgesList = {6, 7, 8, 9, 10, 1, 2, 3, 11, 12, 13, 14, 15, 16, 17, 18};
Field[3].NNodesByEdge = 200;
Field[4] = Threshold;
Field[4].IField = 3;
Field[4].DistMax = 0.5 * rc;
Field[4].DistMin = 0.3 * rc;
Field[4].LcMax = cl__med;
Field[4].LcMin = 0.2 * rc;
// background field: minimum of the fields 1, 2, 4
Field[5] = Min;
Field[5].FieldsList = {1, 2, 4};
Background Field = 5;

Mesh.CharacteristicLengthExtendFromBoundary = 0;

