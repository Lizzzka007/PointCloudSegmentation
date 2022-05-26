/*  CONFORM.H

    Vector Library Include File
    The 3-dim conformity transformations class.

    Copyright (c) 1993 by Octopus Ltd.
    All Rights Reserved.
*/

#ifndef __CONFORM_H
#define __CONFORM_H

#ifndef __cplusplus
#error C++ compiler required
#endif

#include "_defs.h"
#include "Real.h"
#include "Vector.h"
#include "Spin.h"
#include "Point.h"
#include "Plane.h"

#pragma pack (push, 4)

class   CONFORM
{
    public:
  // -------- Implementation -------- //
        SPIN    spin;
        VECTOR3 trans;
        REAL    magn;
  // -------------------------------- //

  // -- Constructor
        CONFORM();
        CONFORM ( const SPIN & , const VECTOR3 & , REAL );

  // -- Operators
        CONFORM operator ~ ( ) const;
        PLANE   operator () ( const PLANE & ) const;

};


// ------------------------------------ //
//  Out-of-class operators & functions  //
// ------------------------------------ //


CONFORM operator >> ( const CONFORM & , const CONFORM & );/*
CONFORM operator >> ( const SPIN & , const CONFORM & );
CONFORM operator >> ( const VECTOR3 & , const CONFORM & );
CONFORM operator >> ( const REAL & , const CONFORM & );
*/

class   BASIS :
public  CONFORM
{
    public:
        int alt;

        BASIS ( );
        BASIS ( const CONFORM & , int = 0 );
        BASIS ( const SPIN &, const VECTOR3 &, REAL , int = 0 );

};

BASIS operator >> ( const BASIS & , const BASIS & );
BASIS operator >> ( const SPIN & , const BASIS & );
BASIS operator >> ( const VECTOR3 & , const BASIS & );
BASIS operator >> ( const REAL & , const BASIS & );

DECLARE const CONFORM CONFORM_1;
DECLARE const BASIS BASIS_1;

#pragma pack (pop)

#endif  //  __CONFORM_H
