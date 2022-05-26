/*  VECTOR2.h

    Geometry Library Include File
    Declarations for 2-dim vectors.

    Copyright (c) 1994 by Octopus Ltd.
    All rights reserved.
*/

#ifndef __VECTOR2_H
#define __VECTOR2_H

#ifndef __cplusplus
#error C++ compiler required
#endif

#include "Real.h"
#include "Vector.h"

#pragma pack (push, 4)

class   VECTOR2
{
  // --------- Data members --------- //
    public:
        REAL    x, y;
  // -------------------------------- //

    public:

  // -- Constructors
        VECTOR2 ( );
        VECTOR2 ( const VECTOR2 & );
        VECTOR2 ( REAL , REAL );

  // -- Indexed access operator
        REAL &          operator [] ( int );
        const REAL &    operator [] ( int ) const;

  // -- Addition & Subtraction operators
        VECTOR2         operator +  ( ) const;
        VECTOR2         operator -  ( ) const;
        VECTOR2 &       operator += ( const VECTOR2 & );
        VECTOR2 &       operator -= ( const VECTOR2 & );

  // -- Multiplication by REAL number operators
        VECTOR2 &       operator *= ( REAL );
        VECTOR2 &       operator /= ( REAL );

  // -- Miscellaneous operators & functions
        REAL    abs ( ) const;
        VECTOR2 fix ( REAL = 1 ) const;
        VECTOR2 project ( const VECTOR2 & ) const;

		double operator* (const VECTOR2 b) { return x*b.x + y*b.y; }

  // -- Out-of-class operators & functions

friend  logical operator == ( const VECTOR2 & , const VECTOR2 & );
friend  logical operator != ( const VECTOR2 & , const VECTOR2 & );
friend  VECTOR2 operator +  ( const VECTOR2 & , const VECTOR2 & );
friend  VECTOR2 operator -  ( const VECTOR2 & , const VECTOR2 & );
friend  VECTOR2 operator *  ( const VECTOR2 & , REAL );
friend  VECTOR2 operator /  ( const VECTOR2 & , REAL );
friend  VECTOR2 operator *  ( REAL , const VECTOR2 & );
friend  REAL    operator |  ( const VECTOR2 & , const VECTOR2 & );
friend  REAL    operator &  ( const VECTOR2 & , const VECTOR2 & );

friend  REAL    abs ( const VECTOR2 & );

};

typedef VECTOR2 POINT2;

extern  const VECTOR2 VECTOR2_0;


// ----------------------------- //
//  Inline function definitions  //
// ----------------------------- //


inline  REAL &  VECTOR2:: operator [] ( int index )
{
    return  ( ( REAL * ) ( void * ) this ) [ index ];
}


inline  const REAL &  VECTOR2:: operator [] ( int index ) const
{
    return  ( ( const REAL * ) ( const void * ) this ) [ index ];
}

struct  POINT_LINK;

extern void xfer3dim2dim ( VECTOR2 * dst, VECTOR3 & ax, VECTOR3 & ay,
        POINT_LINK * src );
extern void xfer3dim2dim ( VECTOR2 * dst, VECTOR3 & ax, VECTOR3 & ay,
        VECTOR3 * v, int n );
extern void xfer3dim2dim ( VECTOR2 * dst, const VECTOR3 & ax, const VECTOR3 & ay,
        const POINT3 * v, const PHINDEX * in );

extern REAL polygonArea ( const VECTOR2 * v, int n );

#pragma pack (pop)

#endif  /* __VECTOR2_H */
