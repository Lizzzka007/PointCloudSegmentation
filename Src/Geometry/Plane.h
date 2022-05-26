/*  PLANE.H

    Vector Library Include File
    Declarations for 3-dim PLANEs.

    Copyright (c) 1993 by Octopus Ltd.
    All Rights Reserved.
*/

#ifndef __PLANE_H
#define __PLANE_H

#ifndef __cplusplus
#error C++ compiler required
#endif

#include "_defs.h"
#include "Real.h"
#include "Vector.h"

#pragma pack (push, 4)

struct  PLANE
{
  // -------- Implementation -------- //
        VECTOR3 norm;
        REAL    dist;
  // -------------------------------- //

  // -- Constructors
        PLANE   ( );
		PLANE(const VECTOR3 &, REAL);

        PLANE   operator -  ( ) const;

        ushort light ( const VECTOR3 & ) const;
};


DECLARE const PLANE PLANE_XY;
DECLARE const PLANE PLANE_YZ;
DECLARE const PLANE PLANE_XZ;

#pragma pack (pop)

#endif  //  __PLANE_H
