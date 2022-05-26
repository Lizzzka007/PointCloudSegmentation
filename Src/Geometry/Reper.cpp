
// #include "StdAfx.h"
#include "math.h"
//#include "../include/vector.h"
#include "Vector.h"

int fsgn ( REAL v ) { return ( v > 0 ) ? 1 : ( ( v < 0 ) ? -1 : 0 ); }

void supplement_reper ( const VECTOR3 & norm, VECTOR3 & ax, VECTOR3 & ay )
{
    int i = 0;
    REAL tmp;
    REAL max = fabs ( norm.x );

    if ( ( tmp = fabs ( norm.y ) > max ) ) { max = tmp; i = 1; }
    if ( ( tmp = fabs ( norm.z ) > max ) ) { max = tmp; i = 2; }
    VECTOR3 base ( VECTOR_0 );
    base [(i+1)%3] = fsgn ( norm [i] );
    ax = ( norm & base ).fix ();
    ay = ( norm & ax );
}
