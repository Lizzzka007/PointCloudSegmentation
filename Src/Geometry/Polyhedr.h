#ifndef __POLYHEDR_H
#define __POLYHEDR_H

/*  POLYHEDR.H

    Polyhedron class definition.

    Copyright (c) 1993 by Octopus Ltd.
    All Rights Reserved.
*/

#ifndef __cplusplus
#error C++ compiler required
#endif

#include <stdio.h>
#include "_defs.h"
#include "Real.h"
#include "Point.h"
#include "Plane.h"
#include "Array.h"
#include "Vector.h"
#include "AllGeometry.h"
#include <vector>

using namespace std;

#pragma pack (push, 4)

struct  PHBONE
{
    PHINDEX   v0, v1, s0, s1;
};

struct PHSIDE
{
    PHINDEX* vlist;
    vector<int> slist;
    vector<int> blist;  
    PLANE plane;

	PHSIDE () { vlist = 0;}

    void copy_vlist(PHINDEX* list, int len)
    {
        this->vlist = new PHINDEX[len];

        for (int i = 0; i < len; ++i)
            this->vlist[i] = list[i];

        return ;
    }

    ~PHSIDE()
    {
        slist.clear();
        blist.clear();
    }
};

int CountSideVertices(const PHINDEX* vlist);

class   BODY;
class   STREAM;
struct  CONVEX_ENVIRONMENT;
struct  OPT_POLYGON2;
struct  OPT_PHBONE;
class   OPT_CUBE;
class   SPECIAL_ARRAY;

class   POLYHEDRON
{
    public:
        int flag;
  // -------- Implementation -------- //
        POINT3  * vertex;
        PHBONE  * bone;
        PHSIDE  * side;
        PHINDEX nverts, nbones, nsides;
        struct convexshell *    shells;
        CONVEX_ENVIRONMENT *    convex_info;
        int *           polygon_info;
        OPT_POLYGON2 *  polygon;
        PLANE *** planemodel;
        OPT_PHBONE * optbone;
        OPT_CUBE * cube_info;
        SPECIAL_ARRAY * cuttings;

  // -------------------------------- //

      // -- Constructors
        POLYHEDRON      ( int nv, int nb, int ns );
        POLYHEDRON      ( const PHINDEX *, int, int, int );
        POLYHEDRON      ( const PHINDEX * );
      // -- Destructor
        virtual         ~ POLYHEDRON      ( );
      //    Other class functions
        CONVEX_ENVIRONMENT * getconvexinfo ( );
        OPT_CUBE *      getoptcube ( );
        void            skipoptcube ( );
        PLANE ***       getplanemodel ( );
        void            prepare2dimsides ( );
        void            skip2dimsides ( );
        OPT_PHBONE *    prepareoptbones ( );
        void            refreshoptbones ( );
        void            initshells () const;
        int             initbones () const;
        void            initsides () const;
        void            cutting ( POLYHEDRON &, POLYHEDRON &, const PLANE & );
        VECTOR3         centerring ( );
        VECTOR3         vertexMassCenter ( );
        logical         possess ( const POINT3 & ) const;
        void            clone ( POLYHEDRON & ) const;
        REAL            volume    () const;

        POLYHEDRON *    retainCuttingPolyhedron ( const PLANE * p, int n );
        void            releaseCuttingPolyhedron ( POLYHEDRON * );
        void            refreshCuttingPolyhedron ( );

        virtual   void  load ( int );
        virtual   int   size ( ) const;

        virtual   int   size ( STREAM * );
        virtual   void  load ( STREAM * );
        virtual   void  save ( STREAM * );
};

struct  convexshell
{
        int n;
        PLANE * planes;
};

extern logical polyhedron_possess ( const POLYHEDRON & octa, const POINT3 & p );

// ---------------------------------------------------- //
// Функция countPlane - Инициализация граней по         //
//   спискам точек ( нормаль, растояние )               //
// ---------------------------------------------------- //

extern void countPlane ( PLANE & plane, const POINT3 * vertex, const PHINDEX * vlist );

#define INFO_MIN 0
#define INFO_MAX 1

#define TEST_S0eqS1  1
#define TEST_S0nrS1  2
#define TEST_VERTfr  4

void test ( const POLYHEDRON & poly, int type, int info );

#pragma pack (pop)

#endif  //  __POLYHEDR_H
