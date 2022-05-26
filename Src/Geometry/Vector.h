/*  VECTOR.H

    Vector Library Include File
    Declarations for 3-dim VECTORs.

    Copyright (c) 1993 by Octopus Ltd.
    All rights reserved.
*/

#ifndef __VECTOR_H
#define __VECTOR_H

#ifndef __cplusplus
#error C++ compiler required
#endif

#include <math.h>
#include "_defs.h"
#include "Real.h"

#pragma pack (push, 4)

struct  VECTOR
{
  // -------- Implementation -------- //
        REAL    x, y, z;
  // -------------------------------- //

  // -- Constructors
        VECTOR  ( );
        VECTOR  ( const VECTOR & );
        VECTOR  ( REAL , REAL , REAL );

  // -- Indexed access operator
        REAL &          operator [] ( int );
        const REAL &    operator [] ( int ) const;

  // -- Addition & Subtraction operators
        VECTOR      operator +  ( ) const;
        VECTOR      operator -  ( ) const;
        VECTOR &    operator += ( const VECTOR & );
        VECTOR &    operator -= ( const VECTOR & );

  // -- Multiplication by REAL number operators
        VECTOR &    operator *= ( REAL );
        VECTOR &    operator /= ( REAL );

  // -- Miscellaneous operators & functions
        REAL        abs ( ) const;
        VECTOR      fix ( REAL = 1 ) const;
        VECTOR      project ( const VECTOR & ) const;

};

typedef VECTOR  VECTOR3;
typedef VECTOR  POINT3;

// ------------------------------------ //
//  Out-of-class operators & functions  //
// ------------------------------------ //

logical operator == ( const VECTOR & , const VECTOR & );
logical operator != ( const VECTOR & , const VECTOR & );
VECTOR  operator +  ( const VECTOR & , const VECTOR & );
VECTOR  operator -  ( const VECTOR & , const VECTOR & );
VECTOR  operator *  ( const VECTOR & , REAL );
VECTOR  operator /  ( const VECTOR & , REAL );
VECTOR  operator *  ( REAL , const VECTOR & );
REAL    operator |  ( const VECTOR & , const VECTOR & );
VECTOR  operator &  ( const VECTOR & , const VECTOR & );

REAL    abs ( const VECTOR & );


DECLARE const VECTOR VECTOR_0;
#define POINT_0 VECTOR_0

// ------------------------------------------------ //
//  Inline definitions for some 'dummy' functions   //
// ------------------------------------------------ //


inline  REAL & VECTOR:: operator [] ( int index )
{
    return  ( ( REAL * ) ( void * ) this ) [ index ];
}


inline  const REAL & VECTOR:: operator [] ( int index ) const
{
    return  ( ( const REAL * ) ( const void * ) this ) [ index ];
}

extern void supplement_reper ( const VECTOR & norm, VECTOR & ax, VECTOR & ay );

#pragma pack (pop)

#endif  //  __VECTOR_H
