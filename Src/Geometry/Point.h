#ifndef __POINT_H
#define __POINT_H

/*  POINT.H

    Vector Library Include File
    Declarations for 3-dim POINTs.

    Copyright (c) 1993 by Octopus Ltd.
    All Rights Reserved.
*/

#ifndef __cplusplus
#error C++ compiler required
#endif

#include "_defs.h"
#include "Real.h"
#include "Vector.h"

#pragma pack (push, 4)

/*
struct  POINT
{
  // -------- Implementation -------- //
        REAL    x, y, z;
  // -------------------------------- //

  // -- Constructors
        POINT   ( );
        POINT   ( const POINT & );
        POINT   ( REAL , REAL , REAL );

  // -- Indexed access operator
        REAL &          operator [] ( int );
        const REAL &    operator [] ( int ) const;

  // -- Type conversions
        VECTOR  radius  ( ) const;

  // -- Operators
        POINT &     operator += ( const VECTOR & );
        POINT &     operator -= ( const VECTOR & );

};


// ------------------------------------ //
//  Out-of-class operators & functions  //
// ------------------------------------ //

VECTOR  operator - ( const POINT & , const POINT & );
POINT   operator + ( const POINT & , const VECTOR & );
POINT   operator - ( const POINT & , const VECTOR & );

VECTOR  radius  ( const POINT & );

DECLARE const POINT3 POINT_0;
*/
#pragma pack (pop)

#endif  //  __POINT_H
