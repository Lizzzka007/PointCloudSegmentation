/*  SPIN.H

    Vector Library Include File
    Declarations for 3-dim SPINs.

    Copyright (c) 1993 by Octopus Ltd.
    All Rights Reserved.
*/

#ifndef __SPIN_H
#define __SPIN_H

#ifndef __cplusplus
#error C++ compiler required
#endif

#include "_defs.h"
#include "Real.h"
#include "Vector.h"
#include "Matrix.h"

#pragma pack (push, 4)

struct  MATRIX;

struct  SPIN
{
    public:
  // -------- Implementation -------- //
        REAL  t, x, y, z;
  // -------------------------------- //

    public:

  // -- Constructors
        SPIN    ( );
        SPIN    ( const SPIN & );
        SPIN    ( const VECTOR3 & , REAL );

  // -- Operators
        SPIN    operator ~  ( ) const;
        VECTOR3 operator () ( const VECTOR3 & ) const;
        MATRIX  operator () ( const MATRIX & ) const;

};


// ------------------------------------ //
//  Out-of-class operators & functions  //
// ------------------------------------ //

SPIN    operator *  ( const SPIN & , const SPIN & );

extern void getspin ( SPIN & s, const VECTOR3 & a, const VECTOR3 & b, const VECTOR3 * n );

DECLARE const SPIN SPIN_1;

#pragma pack (pop)

#endif  //  __SPIN_H
