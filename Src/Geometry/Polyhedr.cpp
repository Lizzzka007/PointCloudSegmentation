
// #include "StdAfx.h"
#include "math.h"
#include "stdio.h"
//#include "../include/Tune.h"
//#include "../include/polyhedr.h"
//#include "../include/stream.h"
#include "Tune.h"
#include "Polyhedr.h"
#include "Stream.h"

POLYHEDRON :: POLYHEDRON ( int n1, int n2, int n3 )
{
	nverts = n1;
	nbones = n2;
	nsides = n3;

  vertex = new POINT3[n1];
  bone   = new PHBONE[n2];
  side   = new PHSIDE[n3];

	flag = 0;
    shells = 0;
	polygon = 0;
	optbone = 0;
	cuttings = 0;
	cube_info = 0;
	planemodel = 0;
	convex_info = 0;
	polygon_info = 0;
}

int length  ( const PHINDEX * list )
{
    PHINDEX first = * list++;
    int len = 2;

    while (first != *list++) 
        len++;
    
    return len;
}

int CountSideVertices(const PHINDEX  * vlist)
{
    if (!vlist)
        return 0;
    return length(vlist) - 1;
}

POLYHEDRON:: POLYHEDRON  ( const PHINDEX * Side, int nv, int nb, int ns )
{
    flag = 0;
    nverts = nv;
    nbones = nb;
    nsides = ns;
    vertex = ( nv > 0 ) ? new POINT3 [nv] : 0;
    bone = ( nb > 0 ) ? new PHBONE [nb] : 0;
    side = ( ns > 0 ) ? new PHSIDE [ns] : 0;
    shells = 0;
    convex_info = 0;
    cube_info = 0;
    polygon_info = 0;
    polygon = 0;
    planemodel = 0;
    optbone = 0;
    cuttings = 0;

    PHINDEX * index;
    for ( int i = 0; i < nsides; i++ )
    {
        int len = length ( Side );
        side [i].vlist = index = new PHINDEX [len];
        while ( len-- > 0 ) * index++ = * Side++;
    }
    nv = initbones  ();
    if ( nv != -1 ) nbones = nv;
}

POLYHEDRON:: POLYHEDRON  ( const PHINDEX * Side )
{
    flag = 0;
    nverts = 0;
    nsides = 0;
    const PHINDEX * s = Side;
    PHINDEX s0;
    while ( * s != MAXPHINDEX )
    {
        s0 = *s;
        do if ( *s >= nverts ) nverts = *s + 1;
        while ( *(++s) != s0 );
        s++;
        nsides++;
    }
    nbones = nsides + nverts - 2;
    vertex = ( nverts > 0 ) ? new POINT3 [nverts] : 0;
    bone = ( nbones > 0 ) ? new PHBONE [nbones] : 0;
    side = ( nsides > 0 ) ? new PHSIDE [nsides] : 0;
    shells = 0;
    convex_info = 0;
    cube_info = 0;
    polygon_info = 0;
    polygon = 0;
    planemodel = 0;
    optbone = 0;
    cuttings = 0;

    PHINDEX * index;
    for ( int i = 0; i < nsides; i++ )
    {
        int len = length ( Side );
        side [i].vlist = index = new PHINDEX [len];
        while ( len-- > 0 ) * index++ = * Side++;
    }
    int nv = initbones  ();
    if ( nv != -1 ) nbones = nv;
}

//extern void makePlaneModel ( PLANE *** & plane, POLYHEDRON & poly );
//extern void skipPlaneModel ( PLANE *** & plane, POLYHEDRON & poly );

void POLYHEDRON_dtor ( POLYHEDRON & p )
{
    int i;
    for ( i = 0; i < p.nsides; i++ )
    {
        delete [] ( PHINDEX * ) p.side[i].vlist;
        p.side[i].slist.clear();  
    }
    if(p.vertex) { delete [] p.vertex; p.vertex = NULL; }
    if(p.bone) { delete [] p.bone; p.bone = NULL; }
    if(p.side) { delete [] p.side; p.side = NULL; }
    p.nsides = 0;
    p.nbones = 0;
    p.nverts = 0;
}

POLYHEDRON:: ~ POLYHEDRON ( )
{
    POLYHEDRON_dtor ( * this );
}

/*
POLYHEDRON :: ~ POLYHEDRON ()
{
    for ( int i = 0; i < nsides; i++ ) delete (void*)side[i].vlist;
//    if ( planemodel ) skipPlaneModel ( planemodel, *this );
	delete vertex;
	delete bone;
	delete side;
}
*/

PLANE *** POLYHEDRON:: getplanemodel ( )
{
//    if ( ! planemodel ) makePlaneModel ( planemodel, * this );
    return planemodel;
}

static logical FoundBone ( PHINDEX & Dst, const PHBONE * List,
                    PHINDEX P1, PHINDEX P2, PHINDEX NSegms )
{
    const PHBONE * Bone;
    for ( int i = 0; i < NSegms; i++ )
    {
        Bone = & List [i];
        if ( ( P1 == Bone -> v0 && P2 == Bone -> v1 ) ||
             ( P2 == Bone -> v0 && P1 == Bone -> v1 ) )
        {
            Dst = i;
            return YES;
        }
    }
    return NO;
}

// void FindAdjacentSides (PHINDEX P, )
// {
//     PHSIDE * p_side = new PHSIDE [nsides];

//     return ;
// }

int POLYHEDRON:: initbones () const
{
    PHINDEX nsegms = 0;
    const PHINDEX * P1, * P2;
    PHINDEX first, index;
    for ( PHINDEX s = 0; s < nsides; s++ )
    {
        const PHINDEX * vlist = side [s].vlist;
        first = * ( P1 = P2 = vlist );
        do
        {
            P2++;
            if ( FoundBone ( index, bone, * P1, * P2, nsegms ) )
            {
                bone [index].s1 = s;
            }
            else
            {
                if ( nsegms == nbones ) return -1;
                PHBONE & Bone = bone [nsegms++];
                Bone.v0 = * P1;
                Bone.v1 = * P2;
                Bone.s0 = Bone.s1 = s;
            }
            P1 = P2;
        }
        while ( * P2 != first );
    }
    return nsegms;
}

void countPlane ( PLANE & plane, const POINT3 * vertex, const PHINDEX * vlist )
{
    int n = 1;
    PHINDEX i0 = * (vlist++);
    const POINT3 & v0 = vertex [ i0 ];
    VECTOR3 sum = v0;
    VECTOR3 norm = VECTOR_0;
    for ( ;; )
    {
        n++;
        sum += vertex [* vlist];
        if ( vlist [1] == i0 ) break;
        norm += ( vertex [ vlist [0] ] - v0 ) & ( vertex [ vlist [1] ] - v0 );
		vlist++;
    }
    sum /= n;
    plane.norm = norm.fix ();
    plane.dist = - ( plane.norm | sum );
}

void    POLYHEDRON:: initsides () const
{
    for ( int i = 0; i < nsides; i++)
    {
        countPlane ( side [i].plane, vertex, side [i].vlist );
    }
}

REAL POLYHEDRON:: volume ( ) const
{
    REAL V = 0;
    for ( int i = 0; i < nsides; i++ )
    {
        const PLANE & Plane = side [i].plane;
        const PHINDEX first = side [i].vlist [0];
        const PHINDEX * v = & side [i].vlist [1];
        const POINT3 & r = vertex [ first ];
        REAL S;
        for ( S = 0; v [1] != first; v++ )
        {
            S += ( ( vertex [ v [0] ] - r ) & ( vertex [ v [1] ] - r ) )
                   | Plane.norm;
        }
        V -= S * Plane.dist;
    }
    return V / 6;
}

void POLYHEDRON:: clone ( POLYHEDRON & p ) const
{
    POLYHEDRON_dtor(p);
    int i, j, len;
    p.vertex = new POINT3 [nverts];
    p.bone = new PHBONE [nbones];
    PHSIDE * p_side = new PHSIDE [nsides];
    for ( i = 0; i < nverts; i++ ) p.vertex [i] = vertex [i];
    for ( i = 0; i < nbones; i++ ) p.bone [i] = bone [i];
    for ( i = 0; i < nsides; i++ )
    {
        p_side [i].plane = side [i].plane;
        len = length ( side [i].vlist );
        PHINDEX * vlist = new PHINDEX [ len ];
        for ( j = 0; j < len; j++ ) vlist [j] = side [i].vlist [j];
        p_side [i].vlist = vlist;
    }
    p.side = p_side;
    p.nverts = nverts;
    p.nbones = nbones;
    p.nsides = nsides;
}

void POLYHEDRON :: load ( int ) {}
int  POLYHEDRON :: size ( ) const { return 0; }

void POLYHEDRON:: save ( STREAM * s )
{
    PHINDEX phLen;
    const PHINDEX * list;
    s -> write ( & nverts, sizeof ( PHINDEX ) );
    s -> write ( & nbones, sizeof ( PHINDEX ) );
    s -> write ( & nsides, sizeof ( PHINDEX ) );
    s -> write ( vertex, sizeof ( POINT3 ) * nverts );
    s -> write ( bone, sizeof ( PHBONE ) * nbones );
    if ( nsides )
    {
        int tmp = 0;
        s -> write ( & tmp, 4 ); // side [0].vlist
    }
    for ( int i = 0; i < nsides; i++ )
    {
        phLen = length ( list = side [i].vlist );
        s -> write ( & phLen, sizeof ( PHINDEX ) );
        s -> write ( list, sizeof ( PHINDEX ) * ( phLen - 1 ) );
    }
    s -> write ( "POL1", 4 );
}
void POLYHEDRON:: load ( STREAM * s )
{
    PHINDEX phLen;
    int len, i, tmp;
    PHINDEX * list;
    POLYHEDRON_dtor ( * this );
    s -> read ( & nverts, sizeof ( PHINDEX ) );
    s -> read ( & nbones, sizeof ( PHINDEX ) );
    s -> read ( & nsides, sizeof ( PHINDEX ) );
    vertex = ( nverts ) ? new POINT3 [nverts] : 0;
    bone =  ( nbones ) ? new PHBONE [nbones] : 0;
    side =  ( nsides ) ? new PHSIDE [nsides] : 0;
    s -> read ( vertex, sizeof ( POINT3 ) * nverts );
    s -> read ( bone, sizeof ( PHBONE ) * nbones );
    logical old = NO;
    if ( nsides )
    {
        s -> read ( & tmp, 4 );
        if ( tmp )
        {
            *((int*)side) = tmp;
            char * mem = ((char*)side) + 4;
            s -> read ( mem, sizeof ( PHSIDE ) - 4 );
            if ( nsides > 1 ) s -> read ( side + 1, sizeof ( PHSIDE ) * ( nsides - 1 ) );
            old = YES;
        }
    }
    for ( i = 0; i < nsides; i++ )
    {
        if ( old )
        {
            s -> read ( & len, sizeof ( int ) );
            side [i].vlist = list = new PHINDEX [len];
            s -> read ( list, sizeof ( PHINDEX ) * len );
        }
        else
        {
            s -> read ( & phLen, sizeof ( PHINDEX ) );
            side [i].vlist = list = new PHINDEX [phLen];
            s -> read ( list, sizeof ( PHINDEX ) * ( phLen - 1 ) );
            list [phLen-1] = list [0];
        }
    }
    if ( ! old )
    {
        s -> read ( & tmp, 4 );
        initsides ();
    }
}

int  POLYHEDRON:: size ( STREAM * s )
{
    int totallen = 3;
    for ( int i = 0; i < nsides; i++ ) totallen += length ( side [i].vlist );
    return
        totallen * sizeof ( PHINDEX ) +                             //   2
        nverts * sizeof ( POINT3 ) +                                //  24
        nbones * sizeof ( PHBONE ) +                                //   8
        4 + ( (nsides) ? 4 : 0 );
}


/*
void test ( const POLYHEDRON & poly, int type, int info )
{
    int i;
    int nseq = 0;
    int npeq = 0;
    int nvfr = 0;
    if ( ( type & TEST_S0eqS1 ) || ( type & TEST_S0nrS1 ) )
    {
        for ( i = 0; i < poly.nbones; ++i )
        {
            const PHBONE & bone = poly.bone[i];
            if ( ( type & TEST_S0eqS1 ) && bone.s0 == bone.s1 )
            {
                ++ nseq;
                if ( info ) message ("s0 == s1");
            }
            else
            if ( ( type & TEST_S0nrS1 ) )
            {
                VECTOR v = poly.side[bone.s0].plane.norm +
                           poly.side[bone.s1].plane.norm;
                double d = poly.side[bone.s0].plane.dist +
                           poly.side[bone.s1].plane.dist;
                if ( micro ( v ) && micro ( d ) )
                {
                    ++ npeq;
                    if ( info ) message ( bone.s0, bone.s1 );
        message(0, length(poly.side[bone.s0].vlist), length(poly.side[bone.s1].vlist));
                }
            }
        }
    }
    if ( ( type & TEST_VERTfr ) )
    {
        Vector3 * vert = ( Vector3 * ) poly.vertex;
        for ( i = 0; i < poly.nsides; ++i )
        {
            const PHINDEX * p = poly.side[i].vlist;
            const Plane & plane = *(Plane*)& poly.side[i].plane;
            int k = 1;
            int j = p[0];
            double d = plane.norm * vert[j] + plane.dist;
            if ( ! micro ( d ) )
            {
                ++ nvfr;
                if ( info )
                    printf ( "vert %d plane %d dist %e\n", j, i, d );
            }
            // Vector3 s ( 0., 0., 0. );
            while ( p[k] != j )
            {
                d = plane.norm * vert[p[k]] + plane.dist;
                if ( ! micro ( d ) )
                {
                    ++ nvfr;
                    if ( info )
                    {
                        char buf[80];
                        sprintf ( buf, "vert %d plane %d dist %e", p[k], i, d );
                        //(!) message(buf);
                    }
                }
                // s += ( vert[p[k-1]] - vert[j] ) % ( vert[p[k]] - vert[j] );
                ++ k;
            }
        // s.norm ();
        // if ( s * plane.norm < 0 ) message("<0");
        // if ( !near_( s, plane.norm ) )
        //   message("no near");
        }
    }
    if ( ( type & TEST_S0eqS1 ) && nseq )
        printf ( "%d equal side in bones\n", nseq );
    if ( ( type & TEST_S0nrS1 ) && npeq )
        printf ( "%d sides is near\n", npeq );
    if ( ( type & TEST_VERTfr ) && nvfr )
        printf ( "%d verts is far\n", nvfr );
}
*/
